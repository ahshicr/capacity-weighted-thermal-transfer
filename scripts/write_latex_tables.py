from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"
METHOD_LABELS = {
    "nearest_flux": "Nearest flux",
    "inverse_distance_flux": "Inverse distance",
    "conservative_local_rate": "Local conservative",
    "conservative_moment_projection": "Moment projection",
}
CASE_LABELS = {
    "gci_fine": "Fine grid reference",
    "z2_forced": "Zone 2 forced",
    "z2_fvdom_realizable": "Zone 2 radiation, realizable $k$-$\\epsilon$",
    "z2_fvdom_rng": "Zone 2 radiation, RNG $k$-$\\epsilon$",
    "z2_fvdom_sst": "Zone 2 radiation, SST",
    "z2_shell": "Zone 2 thermal shell",
    "z5_buoyant": "Zone 5 buoyant",
    "z5_forced": "Zone 5 forced",
    "z5_fvdom": "Zone 5 radiation",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_cases()
    write_methods()
    write_gci()
    write_sensitivity()


def write_cases() -> None:
    frame = pd.read_csv(ROOT / "results" / "transfer_analysis" / "case_summary.csv")
    lines = []
    for row in frame.itertuples():
        lines.append(
            f"{CASE_LABELS[row.case]} & {row.face_count:,} & {row.surface_area_m2:.3f} & "
            f"{row.wall_heat_flux_mean_W_m2:.1f} & {row.projection_condition_number:.2f} & "
            f"{100.0 * row.projection_relative_correction:.2f} \\\\"
        )
    write_rows("case_rows.tex", lines)


def write_methods() -> None:
    frame = pd.read_csv(ROOT / "results" / "transfer_analysis" / "aggregate_metrics.csv").set_index("method")
    lines = []
    for method, label in METHOD_LABELS.items():
        row = frame.loc[method]
        lines.append(
            f"{label} & {100.0 * row.relative_heat_error_median:.3g} & "
            f"{100.0 * row.relative_first_moment_error_median:.3g} & "
            f"{row.relative_back_projection_rmse_median:.3f} & "
            f"{100.0 * row.relative_final_temperature_change_median:.2f} & "
            f"{row.normalized_graph_roughness_median:.3f} \\\\"
        )
    write_rows("method_rows.tex", lines)


def write_gci() -> None:
    frame = pd.read_csv(ROOT / "results" / "gci_transfer" / "gci_quantities.csv")
    labels = {
        "area_mean_wall_heat_flux_W_m2": "Area mean wall heat flux",
        "mapped_total_heat_rate_W": "Mapped total heat rate",
        "mapped_nodal_rate_rms_W": "Mapped nodal rate RMS",
        "mapped_nodal_rate_p95_abs_W": "Mapped nodal rate P95",
    }
    lines = []
    for row in frame.itertuples():
        lines.append(
            f"{labels[row.quantity]} & {row.observed_order:.3f} & {row.fine_gci_percent:.3f} & "
            f"{row.medium_gci_percent:.3f} & {row.asymptotic_ratio:.3f} \\\\"
        )
    write_rows("gci_rows.tex", lines)


def write_sensitivity() -> None:
    frame = pd.read_csv(ROOT / "results" / "analysis" / "neighborhood_sensitivity.csv")
    frame = frame[frame.neighbors.isin([2, 4, 8, 12, 24, 64])]
    lines = []
    for row in frame.itertuples():
        marker = "yes" if bool(row.selected) else "no"
        lines.append(
            f"{row.neighbors} & {row.relative_back_projection_rmse_median:.3f} & "
            f"{100.0 * row.relative_final_temperature_change_median:.2f} & "
            f"{row.normalized_graph_roughness_median:.3f} & {marker} \\\\"
        )
    write_rows("sensitivity_rows.tex", lines)


def write_rows(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n\\bottomrule\n", encoding="ascii")


if __name__ == "__main__":
    main()
