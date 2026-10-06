from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from analyze_transfer import _analyze_case, _load_laplacian, _load_targets
from vtp_polydata import read_ascii_vtp_surface


CELL_COUNTS = {
    "coarse": 1_585_775,
    "medium": 3_553_009,
    "fine": 7_891_695,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grid convergence of mapped full-surface heat loads.")
    parser.add_argument(
        "--coarse",
        type=Path,
        default=ROOT / "vehicle_surface_fields" / "wall_fields" / "gci_coarse",
    )
    parser.add_argument(
        "--medium",
        type=Path,
        default=ROOT / "vehicle_surface_fields" / "wall_fields" / "gci_medium",
    )
    parser.add_argument(
        "--fine",
        type=Path,
        default=ROOT / "vehicle_surface_fields" / "wall_fields" / "gci_fine",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gci_transfer")
    args = parser.parse_args()

    targets = _load_targets(
        ROOT / "data" / "vehicle" / "thermal_nodes.csv",
        ROOT / "data" / "vehicle" / "body_envelope_points_40mm.npz",
    )
    laplacian = _load_laplacian(ROOT / "data" / "vehicle" / "thermal_links.csv", targets["node_id"])
    mesh_directories = {"coarse": args.coarse, "medium": args.medium, "fine": args.fine}
    mapped: dict[str, np.ndarray] = {}
    source_statistics: dict[str, dict[str, float]] = {}
    for level, directory in mesh_directories.items():
        path = next(directory.rglob("*.vtp"))
        surface = read_ascii_vtp_surface(path)
        _, summary, representative = _analyze_case(
            level,
            surface,
            targets,
            laplacian,
            neighbors=8,
            workers=-1,
            projection_metric="capacity",
            duration_s=300.0,
            time_step_s=5.0,
        )
        mapped[level] = representative["conservative_moment_projection_rate_W"].astype(float)
        source_statistics[level] = summary

    quantities = {
        "area_mean_wall_heat_flux_W_m2": np.array(
            [source_statistics[level]["wall_heat_flux_mean_W_m2"] for level in ("coarse", "medium", "fine")]
        ),
        "mapped_total_heat_rate_W": np.array(
            [mapped[level].sum() for level in ("coarse", "medium", "fine")]
        ),
        "mapped_nodal_rate_rms_W": np.array(
            [np.sqrt(np.mean(mapped[level] ** 2)) for level in ("coarse", "medium", "fine")]
        ),
        "mapped_nodal_rate_p95_abs_W": np.array(
            [np.quantile(np.abs(mapped[level]), 0.95) for level in ("coarse", "medium", "fine")]
        ),
    }
    rows = []
    for name, coarse_medium_fine in quantities.items():
        result = _gci_three_grid(*coarse_medium_fine)
        rows.append({"quantity": name, **result})

    node_rows = []
    for node, node_id in enumerate(targets["node_id"]):
        values = np.array([mapped[level][node] for level in ("coarse", "medium", "fine")])
        try:
            result = _gci_three_grid(*values)
        except ValueError:
            continue
        node_rows.append({"node_id": node_id, "body_region": targets["region"][node], **result})

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "gci_quantities.csv", index=False)
    node_frame = pd.DataFrame(node_rows)
    node_frame.to_csv(args.output / "gci_nodewise.csv", index=False)
    summary = {
        "cell_counts": CELL_COUNTS,
        "mapped_node_count": len(targets["node_id"]),
        "nodewise_valid_count": len(node_frame),
        "nodewise_fine_gci_median_percent": float(node_frame["fine_gci_percent"].median()),
        "nodewise_fine_gci_p90_percent": float(node_frame["fine_gci_percent"].quantile(0.90)),
        "nodewise_fine_gci_p95_percent": float(node_frame["fine_gci_percent"].quantile(0.95)),
        "nodewise_fine_gci_max_percent": float(node_frame["fine_gci_percent"].max()),
    }
    (args.output / "gci_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


def _gci_three_grid(coarse: float, medium: float, fine: float) -> dict[str, float]:
    h_coarse = CELL_COUNTS["coarse"] ** (-1.0 / 3.0)
    h_medium = CELL_COUNTS["medium"] ** (-1.0 / 3.0)
    h_fine = CELL_COUNTS["fine"] ** (-1.0 / 3.0)
    r21 = h_medium / h_fine
    r32 = h_coarse / h_medium
    epsilon21 = medium - fine
    epsilon32 = coarse - medium
    if abs(fine) < 1.0e-14 or abs(epsilon21) < 1.0e-14 or abs(epsilon32) < 1.0e-14:
        raise ValueError("degenerate three-grid sequence")
    sign = 1.0 if epsilon32 / epsilon21 >= 0.0 else -1.0

    def equation(order: float) -> float:
        numerator = r21**order - sign
        denominator = r32**order - sign
        if numerator <= 0.0 or denominator <= 0.0:
            return np.nan
        return order - abs(np.log(abs(epsilon32 / epsilon21)) + np.log(numerator / denominator)) / np.log(r21)

    grid = np.linspace(0.05, 12.0, 600)
    roots = []
    previous_x = grid[0]
    previous_y = equation(previous_x)
    for current_x in grid[1:]:
        current_y = equation(current_x)
        if np.isfinite(previous_y) and np.isfinite(current_y) and previous_y * current_y < 0.0:
            roots.append(brentq(equation, previous_x, current_x))
        previous_x, previous_y = current_x, current_y
    if not roots:
        raise ValueError("no observed-order root in the admissible range")
    order = min(roots, key=lambda value: abs(value - 2.0))
    extrapolated = (r21**order * fine - medium) / (r21**order - 1.0)
    approximate_fine = abs((fine - medium) / fine)
    approximate_medium = abs((medium - coarse) / medium)
    fine_gci = 1.25 * approximate_fine / (r21**order - 1.0)
    medium_gci = 1.25 * approximate_medium / (r32**order - 1.0)
    ratio = medium_gci / max(r21**order * fine_gci, 1.0e-30)
    return {
        "coarse_value": coarse,
        "medium_value": medium,
        "fine_value": fine,
        "observed_order": order,
        "extrapolated_value": extrapolated,
        "fine_gci_percent": 100.0 * fine_gci,
        "medium_gci_percent": 100.0 * medium_gci,
        "asymptotic_ratio": ratio,
    }


if __name__ == "__main__":
    main()
