"""Run frozen benchmark comparisons across one or more experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cfd_operator.config import load_config
from cfd_operator.evaluators import run_split_evaluation
from cfd_operator.visualization import (
    plot_gap_bars,
    plot_multi_experiment_metric_comparison,
    plot_split_metric_bars,
)


def _compute_generalization_gaps(test_metrics: dict[str, float], benchmark_metrics: dict[str, float]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    if "field_rmse" in test_metrics and "field_rmse" in benchmark_metrics:
        gaps["benchmark_holdout_field_gap"] = float(benchmark_metrics["field_rmse"] - test_metrics["field_rmse"])
    scalar_components = []
    for key in ("cl_mae", "cd_mae"):
        if key in test_metrics and key in benchmark_metrics:
            scalar_components.append(float(benchmark_metrics[key] - test_metrics[key]))
    if scalar_components:
        gaps["benchmark_holdout_scalar_gap"] = float(sum(scalar_components) / len(scalar_components))
    if "cp_surface_rmse" in test_metrics and "cp_surface_rmse" in benchmark_metrics:
        gaps["benchmark_holdout_surface_gap"] = float(benchmark_metrics["cp_surface_rmse"] - test_metrics["cp_surface_rmse"])
    if "slice_rmse" in test_metrics and "slice_rmse" in benchmark_metrics:
        gaps["benchmark_holdout_slice_gap"] = float(benchmark_metrics["slice_rmse"] - test_metrics["slice_rmse"])
    feature_components = []
    for key in ("high_gradient_iou", "pressure_gradient_indicator_f1"):
        if key in test_metrics and key in benchmark_metrics:
            feature_components.append(float(test_metrics[key] - benchmark_metrics[key]))
    if feature_components:
        gaps["benchmark_holdout_feature_gap"] = float(sum(feature_components) / len(feature_components))
    return gaps


def _summary_rows(run_name: str, test_metrics: dict[str, float], benchmark_metrics: dict[str, float], gaps: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_metric_names = sorted(set(test_metrics.keys()) | set(benchmark_metrics.keys()))
    for metric_name in all_metric_names:
        rows.append(
            {
                "run_name": run_name,
                "metric_name": metric_name,
                "test_value": test_metrics.get(metric_name),
                "benchmark_holdout_value": benchmark_metrics.get(metric_name),
            }
        )
    for gap_name, gap_value in gaps.items():
        rows.append(
            {
                "run_name": run_name,
                "metric_name": gap_name,
                "test_value": None,
                "benchmark_holdout_value": float(gap_value),
            }
        )
    return rows


def _write_markdown(output_path: Path, run_results: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        "## Benchmark Protocol",
        "",
        "- `test` is the development split inside the 90% main pool.",
        "- `benchmark_holdout` is the frozen benchmark split and must not be used for training, early stopping, or model selection.",
        "- Generalization/robustness gaps are computed from `test` and `benchmark_holdout` results only.",
        "",
    ]
    for result in run_results:
        lines.extend(
            [
                f"## {result['run_name']}",
                "",
                "| Metric | test | benchmark_holdout |",
                "| --- | ---: | ---: |",
            ]
        )
        all_metric_names = sorted(set(result["test_metrics"].keys()) | set(result["benchmark_metrics"].keys()))
        for metric_name in all_metric_names:
            lines.append(
                f"| {metric_name} | {result['test_metrics'].get(metric_name, 'n/a')} | {result['benchmark_metrics'].get(metric_name, 'n/a')} |"
            )
        lines.extend(["", "### Generalization / Robustness Gap", "", "| Gap | Value |", "| --- | ---: |"])
        for gap_name, gap_value in result["gaps"].items():
            lines.append(f"| {gap_name} | {gap_value} |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test vs benchmark_holdout comparisons for one or more experiments.")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("RUN_NAME", "CONFIG", "CHECKPOINT"),
        required=True,
        help="Add a benchmark run as: --run <name> <config> <checkpoint>",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run_name, config_path, checkpoint_path in args.run:
        config = load_config(config_path, overrides=[])
        if args.device is not None:
            config.experiment.device = args.device
        run_dir = output_dir / run_name
        test_result = run_split_evaluation(
            config=config,
            checkpoint_path=checkpoint_path,
            split_name="test",
            output_dir=run_dir / "test",
        )
        benchmark_result = run_split_evaluation(
            config=config,
            checkpoint_path=checkpoint_path,
            split_name="benchmark_holdout",
            output_dir=run_dir / "benchmark_holdout",
        )
        test_metrics = test_result["metrics"]
        benchmark_metrics = benchmark_result["metrics"]
        assert isinstance(test_metrics, dict)
        assert isinstance(benchmark_metrics, dict)
        gaps = _compute_generalization_gaps(test_metrics, benchmark_metrics)
        run_results.append(
            {
                "run_name": run_name,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "test_metrics": test_metrics,
                "benchmark_metrics": benchmark_metrics,
                "gaps": gaps,
                "test_output_dir": test_result["output_dir"],
                "benchmark_output_dir": benchmark_result["output_dir"],
            }
        )
        all_rows.extend(_summary_rows(run_name, test_metrics, benchmark_metrics, gaps))

        key_metrics = {
            "field_rmse": "Field RMSE",
            "cl_mae": "Cl MAE",
            "cd_mae": "Cd MAE",
            "cp_surface_rmse": "Cp Surface RMSE",
            "slice_rmse": "Slice RMSE",
        }
        for metric_name, title in key_metrics.items():
            if metric_name not in test_metrics or metric_name not in benchmark_metrics:
                continue
            plot_split_metric_bars(
                {
                    "test": float(test_metrics[metric_name]),
                    "benchmark_holdout": float(benchmark_metrics[metric_name]),
                },
                title=f"{run_name}: {title}",
                ylabel=metric_name,
                save_path=run_dir / f"{metric_name}_split_comparison.png",
            )
        if gaps:
            plot_gap_bars(
                gaps,
                title=f"{run_name}: test -> benchmark_holdout gaps",
                save_path=run_dir / "generalization_gap.png",
            )

    summary_json = output_dir / "benchmark_summary.json"
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_md = output_dir / "benchmark_summary.md"
    summary_json.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
    _write_markdown(summary_md, run_results)

    comparison_metrics = ["field_rmse", "cl_mae", "cd_mae", "cp_surface_rmse", "slice_rmse"]
    for metric_name in comparison_metrics:
        experiment_names = []
        test_values = []
        benchmark_values = []
        for result in run_results:
            if metric_name not in result["test_metrics"] or metric_name not in result["benchmark_metrics"]:
                continue
            experiment_names.append(str(result["run_name"]))
            test_values.append(float(result["test_metrics"][metric_name]))
            benchmark_values.append(float(result["benchmark_metrics"][metric_name]))
        if experiment_names:
            plot_multi_experiment_metric_comparison(
                experiment_names=experiment_names,
                test_values=test_values,
                benchmark_values=benchmark_values,
                metric_name=metric_name,
                save_path=output_dir / f"multi_experiment_{metric_name}.png",
            )

    print(f"Saved benchmark summary to {summary_json}")
    print(f"Saved benchmark table to {summary_csv}")
    print(f"Saved benchmark report to {summary_md}")


if __name__ == "__main__":
    main()
