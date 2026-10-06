from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROPOSED = "conservative_moment_projection"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate transfer, sensitivity, and GCI evidence.")
    parser.add_argument("--transfer", type=Path, default=ROOT / "results" / "transfer_analysis")
    parser.add_argument("--sweep", type=Path, default=ROOT / "results" / "transfer_sweep")
    parser.add_argument("--gci", type=Path, default=ROOT / "results" / "gci_transfer")
    parser.add_argument("--global-gci", type=Path, default=ROOT / "data" / "gci" / "grid_convergence.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    transfer = pd.read_csv(args.transfer / "transfer_metrics.csv")
    sweep = pd.read_csv(args.sweep / "sweep_aggregate_metrics.csv")
    gci = pd.read_csv(args.gci / "gci_quantities.csv")
    global_gci = json.loads(args.global_gci.read_text(encoding="utf-8"))
    validate_transfer(transfer)

    comparison = paired_comparisons(transfer)
    sensitivity = select_neighborhood(sweep)
    metric = compare_projection_metrics(sweep)
    summary = build_summary(transfer, sensitivity, gci, global_gci)

    comparison.to_csv(args.output / "paired_transfer_comparisons.csv", index=False)
    sensitivity.to_csv(args.output / "neighborhood_sensitivity.csv", index=False)
    metric.to_csv(args.output / "projection_metric_comparison.csv", index=False)
    gci.to_csv(args.output / "gci_table.csv", index=False)
    (args.output / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def validate_transfer(frame: pd.DataFrame) -> None:
    expected_methods = {
        "nearest_flux",
        "inverse_distance_flux",
        "conservative_local_rate",
        PROPOSED,
    }
    if set(frame.method) != expected_methods:
        raise RuntimeError("the transfer comparison does not contain the declared methods")
    case_counts = frame.groupby("method").case.nunique()
    if not np.all(case_counts.to_numpy() == 9):
        raise RuntimeError("each transfer method must be evaluated on all nine CFD cases")


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "relative_heat_error",
        "relative_first_moment_error",
        "relative_back_projection_rmse",
        "relative_final_temperature_change",
        "normalized_graph_roughness",
    )
    proposed = frame[frame.method == PROPOSED].set_index("case")
    rows: list[dict] = []
    for baseline_name in sorted(set(frame.method) - {PROPOSED}):
        baseline = frame[frame.method == baseline_name].set_index("case")
        for metric in metrics:
            p_values = proposed.loc[baseline.index, metric].to_numpy(dtype=float)
            b_values = baseline[metric].to_numpy(dtype=float)
            difference = p_values - b_values
            rows.append(
                {
                    "metric": metric,
                    "baseline_method": baseline_name,
                    "case_count": int(difference.size),
                    "proposed_median": float(np.median(p_values)),
                    "baseline_median": float(np.median(b_values)),
                    "median_difference": float(np.median(difference)),
                    "mean_difference": float(np.mean(difference)),
                    "exact_sign_flip_p": exact_sign_flip_p(difference),
                }
            )
    return pd.DataFrame(rows)


def select_neighborhood(sweep: pd.DataFrame) -> pd.DataFrame:
    frame = sweep[
        (sweep.method == PROPOSED)
        & (sweep.projection_metric == "capacity")
    ].copy()
    frame = frame.sort_values("neighbors")
    frame["admissible"] = (
        (frame.relative_back_projection_rmse_median <= 0.14)
        & (frame.relative_final_temperature_change_median <= 0.02)
        & (frame.normalized_graph_roughness_median <= 0.16)
    )
    admissible_neighbors = frame.loc[frame.admissible, "neighbors"]
    if admissible_neighbors.empty:
        raise RuntimeError("no neighborhood satisfies the declared accuracy and smoothness envelope")
    selected_neighbors = int(admissible_neighbors.min())
    frame["selected"] = frame.neighbors == selected_neighbors
    columns = [
        "neighbors",
        "projection_metric",
        "relative_first_moment_error_median",
        "relative_back_projection_rmse_median",
        "relative_rate_change_from_local_median",
        "relative_final_temperature_change_median",
        "normalized_graph_roughness_median",
        "admissible",
        "selected",
    ]
    return frame[columns]
def compare_projection_metrics(sweep: pd.DataFrame) -> pd.DataFrame:
    return sweep[
        (sweep.neighbors == 24)
        & (sweep.method == PROPOSED)
        & (sweep.projection_metric.isin(["capacity", "uniform", "surface_area"]))
    ][
        [
            "projection_metric",
            "relative_first_moment_error_median",
            "relative_rate_change_from_local_median",
            "relative_back_projection_rmse_median",
            "relative_final_temperature_change_median",
            "normalized_graph_roughness_median",
        ]
    ].sort_values("projection_metric")


def build_summary(
    transfer: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gci: pd.DataFrame,
    global_gci: dict,
) -> dict:
    proposed = transfer[transfer.method == PROPOSED]
    selected = sensitivity[sensitivity.selected].iloc[0]
    return {
        "case_count": int(transfer.case.nunique()),
        "source_face_count_min": int(transfer.source_faces.min()),
        "source_face_count_max": int(transfer.source_faces.max()),
        "target_node_count": 421,
        "selected_neighbors": int(selected.neighbors),
        "proposed_relative_heat_error_max": float(proposed.relative_heat_error.max()),
        "proposed_relative_first_moment_error_max": float(proposed.relative_first_moment_error.max()),
        "proposed_relative_back_projection_rmse_median": float(proposed.relative_back_projection_rmse.median()),
        "proposed_relative_final_temperature_change_median": float(proposed.relative_final_temperature_change.median()),
        "proposed_normalized_graph_roughness_median": float(proposed.normalized_graph_roughness.median()),
        "global_tail_fine_gci_percent": float(global_gci["fine_grid_GCI_percent"]),
        "global_tail_observed_order": float(global_gci["observed_order"]),
        "global_tail_asymptotic_ratio": float(global_gci["asymptotic_ratio"]),
        "field_level_gci": {
            row.quantity: {
                "fine_gci_percent": float(row.fine_gci_percent),
                "observed_order": float(row.observed_order),
                "asymptotic_ratio": float(row.asymptotic_ratio),
            }
            for row in gci.itertuples()
        },
    }


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1.0e-15))


if __name__ == "__main__":
    main()
