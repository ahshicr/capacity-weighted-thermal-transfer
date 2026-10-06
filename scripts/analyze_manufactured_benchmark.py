from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "manufactured_benchmark"
TABLES = ROOT / "tables"

METHOD_ORDER = [
    "nearest",
    "inverse_distance",
    "local_rbf",
    "total_scaled_rbf",
    "rbf_uniform_moment",
    "rbf_capacity_moment",
    "common_refinement",
]

METHOD_LABELS = {
    "nearest": "Nearest",
    "inverse_distance": "Inverse distance",
    "local_rbf": "Local RBF",
    "total_scaled_rbf": "Total-scaled RBF",
    "rbf_uniform_moment": "RBF with uniform projection",
    "rbf_capacity_moment": "RBF with capacity projection",
    "common_refinement": "Common refinement",
    "local_conservative": "Local conservative distribution",
    "uniform_moment": "Local distribution with uniform projection",
    "capacity_moment": "Local distribution with capacity projection",
}


def paired(frame: pd.DataFrame, first: str, second: str, metric: str) -> dict[str, float | int | str]:
    index = ["source_nx", "source_ny", "target_nx", "target_ny", "field"]
    left = frame[frame.method == first].set_index(index)[metric]
    right = frame[frame.method == second].set_index(index)[metric]
    joined = pd.concat([left.rename("first"), right.rename("second")], axis=1, join="inner").dropna()
    if not np.array_equal(joined.index.to_numpy(), joined.index.to_numpy()):
        raise RuntimeError("paired benchmark indices are inconsistent")
    differences = joined["first"].to_numpy() - joined["second"].to_numpy()
    if np.allclose(differences, 0.0, rtol=0.0, atol=1.0e-30):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(joined["first"], joined["second"], alternative="two-sided").pvalue)
    ratios = joined["first"].to_numpy() / np.maximum(joined["second"].to_numpy(), 1.0e-30)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "pairs": int(len(joined)),
        "first_wins": int(np.sum(joined["first"].to_numpy() < joined["second"].to_numpy())),
        "ties": int(np.sum(np.isclose(joined["first"], joined["second"], rtol=1.0e-12, atol=1.0e-16))),
        "median_ratio": float(np.median(ratios)),
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(np.maximum(ratios, 1.0e-30))))),
        "wilcoxon_p": p_value,
    }


def holm_adjust(rows: list[dict[str, float | int | str]]) -> None:
    order = np.argsort([float(row["wilcoxon_p"]) for row in rows])
    count = len(rows)
    running = 0.0
    for rank, position in enumerate(order):
        raw = float(rows[position]["wilcoxon_p"])
        adjusted = min(1.0, (count - rank) * raw)
        running = max(running, adjusted)
        rows[position]["holm_adjusted_p"] = running


def write_accuracy_table(frame: pd.DataFrame) -> None:
    lines: list[str] = []
    for method in METHOD_ORDER:
        current = frame[frame.method == method]
        load = 100.0 * current.load_l2_relative.to_numpy()
        thermal = 100.0 * current.thermal_rmse_relative.to_numpy()
        lines.append(
            f"{METHOD_LABELS[method]} & {np.median(load):.3f} & "
            f"{np.quantile(load, 0.25):.3f} to {np.quantile(load, 0.75):.3f} & "
            f"{np.median(thermal):.3f} & {current.source_total_residual.max():.2e} & "
            f"{current.source_moment_residual.max():.2e} \\\\"
        )
    (TABLES / "manufactured_accuracy_rows.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n", encoding="ascii"
    )


def write_energy_table(frame: pd.DataFrame) -> None:
    lines: list[str] = []
    for method in ("total_scaled_rbf", "rbf_uniform_moment", "rbf_capacity_moment"):
        current = frame[frame.method == method]
        correction = 100.0 * current.rbf_correction_energy_relative.to_numpy()
        response = 100.0 * current.rbf_thermal_deviation_relative.to_numpy()
        lines.append(
            f"{METHOD_LABELS[method]} & {np.median(correction):.4f} & "
            f"{np.quantile(correction, 0.75):.4f} & {np.median(response):.4f} & "
            f"{np.quantile(response, 0.75):.4f} \\\\"
        )
    (TABLES / "projection_energy_rows.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n", encoding="ascii"
    )


def write_scaling_table(scaling: pd.DataFrame) -> None:
    lines: list[str] = []
    for row in scaling.itertuples():
        lines.append(
            f"{row.source_cells:,} & {1000.0 * row.median_runtime_s:.2f} & "
            f"{row.estimated_matrix_free_memory_mb:.1f} & {row.dense_operator_memory_mb:.1f} & "
            f"{row.maximum_total_residual:.2e} & {row.maximum_moment_residual:.2e} \\\\"
        )
    (TABLES / "scaling_rows.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n", encoding="ascii"
    )


def write_field_table(frame: pd.DataFrame) -> None:
    lines: list[str] = []
    for field in ("linear", "smooth_hotspot", "thermal_front", "multiscale"):
        proposed = frame[(frame.method == "rbf_capacity_moment") & (frame.field == field)]
        common = frame[(frame.method == "common_refinement") & (frame.field == field)]
        lines.append(
            f"{field.replace('_', ' ').title()} & {100.0 * proposed.load_l2_relative.median():.3f} & "
            f"{100.0 * proposed.thermal_rmse_relative.median():.3f} & "
            f"{100.0 * common.load_l2_relative.median():.3f} & "
            f"{100.0 * common.thermal_rmse_relative.median():.3f} \\\\"
        )
    (TABLES / "field_accuracy_rows.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\n", encoding="ascii"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS / "accuracy_metrics.csv")
    scaling = pd.read_csv(RESULTS / "runtime_scaling.csv")
    manifest = json.loads((RESULTS / "benchmark_manifest.json").read_text(encoding="utf-8"))
    expected = 48
    counts = frame.groupby("method").size()
    if not (counts == expected).all():
        raise RuntimeError(f"incomplete benchmark: {counts.to_dict()}")
    comparisons = [
        paired(frame, "rbf_capacity_moment", method, "load_l2_relative")
        for method in METHOD_ORDER
        if method != "rbf_capacity_moment"
    ]
    comparisons += [
        paired(frame, "rbf_capacity_moment", method, "thermal_rmse_relative")
        for method in METHOD_ORDER
        if method != "rbf_capacity_moment"
    ]
    comparisons += [
        paired(
            frame,
            "rbf_capacity_moment",
            "rbf_uniform_moment",
            "rbf_correction_energy_relative",
        ),
        paired(
            frame,
            "rbf_capacity_moment",
            "rbf_uniform_moment",
            "rbf_thermal_deviation_relative",
        ),
    ]
    holm_adjust(comparisons)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(RESULTS / "paired_comparisons.csv", index=False)

    capacity = frame[frame.method == "rbf_capacity_moment"]
    uniform = frame[frame.method == "rbf_uniform_moment"]
    summary = {
        "physical_scenarios": expected,
        "method_results": int(len(frame)),
        "maximum_capacity_projection_total_residual": float(capacity.source_total_residual.max()),
        "maximum_capacity_projection_moment_residual": float(capacity.source_moment_residual.max()),
        "maximum_capacity_projection_negative_fraction": float(capacity.negative_load_fraction.max()),
        "median_capacity_projection_load_error_percent": float(100.0 * capacity.load_l2_relative.median()),
        "median_capacity_projection_thermal_error_percent": float(100.0 * capacity.thermal_rmse_relative.median()),
        "capacity_energy_lower_cases": int(
            np.sum(
                capacity.rbf_correction_energy_relative.to_numpy()
                <= uniform.rbf_correction_energy_relative.to_numpy() * (1.0 + 1.0e-12)
            )
        ),
        "quadrature_maximum_relative_difference": float(
            manifest["quadrature_audit"]["maximum_relative_difference"]
        ),
        "common_refinement_maximum_coverage_error": float(frame.coverage_error.max()),
        "million_source_runtime_s": float(scaling.iloc[-1].median_runtime_s),
        "million_source_matrix_free_memory_mb": float(scaling.iloc[-1].estimated_matrix_free_memory_mb),
        "million_source_dense_memory_mb": float(scaling.iloc[-1].dense_operator_memory_mb),
        "comparisons": comparisons,
    }
    (RESULTS / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_accuracy_table(frame)
    write_energy_table(frame)
    write_scaling_table(scaling)
    write_field_table(frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
