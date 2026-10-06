from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from conservative_transfer import moment_matrices, scaled_coordinates
from vtp_polydata import SurfaceField, read_ascii_vtp_surface


METHODS = (
    "nearest_flux",
    "inverse_distance_flux",
    "conservative_local_rate",
    "conservative_moment_projection",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conservative thermal transfer on complete CFD surfaces.")
    parser.add_argument(
        "--surface-root",
        type=Path,
        default=ROOT / "vehicle_surface_fields" / "wall_fields",
    )
    parser.add_argument("--nodes", type=Path, default=ROOT / "data" / "vehicle" / "thermal_nodes.csv")
    parser.add_argument("--links", type=Path, default=ROOT / "data" / "vehicle" / "thermal_links.csv")
    parser.add_argument(
        "--envelope",
        type=Path,
        default=ROOT / "data" / "vehicle" / "body_envelope_points_40mm.npz",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "transfer_analysis")
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--projection-metric",
        choices=("capacity", "uniform", "surface_area"),
        default="capacity",
    )
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--time-step-s", type=float, default=5.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(args.nodes, args.envelope)
    laplacian = _load_laplacian(args.links, targets["node_id"])
    # The supplementary archive also contains two GCI-only grids.
    # Exclude these from the nine-case vehicle application.
    paths = [
        path for path in sorted(args.surface_root.rglob("*.vtp"))
        if _case_label(path, args.surface_root) not in {"gci_coarse", "gci_medium"}
    ]
    if not paths:
        raise FileNotFoundError(f"no VTP surface files found under {args.surface_root}")

    rows: list[dict] = []
    cases: list[dict] = []
    started = time.perf_counter()
    representative_written = False
    for index, path in enumerate(paths, start=1):
        label = _case_label(path, args.surface_root)
        surface = read_ascii_vtp_surface(path)
        if "wallHeatFlux" not in surface.cell_data:
            raise ValueError(f"wallHeatFlux is absent from {path}")
        current_rows, case_summary, representative = _analyze_case(
            label,
            surface,
            targets,
            laplacian,
            neighbors=args.neighbors,
            workers=args.workers,
            projection_metric=args.projection_metric,
            duration_s=args.duration_s,
            time_step_s=args.time_step_s,
        )
        rows.extend(current_rows)
        case_summary.update(
            {
                "source_path": path.relative_to(args.surface_root).as_posix(),
                "source_sha256": _sha256(path),
                "source_bytes": path.stat().st_size,
            }
        )
        cases.append(case_summary)
        if not representative_written and label == "z2_fvdom_sst":
            np.savez_compressed(args.output / "representative_surface.npz", **representative)
            representative_written = True
        print(
            f"{index:02d}/{len(paths):02d} {label:22s} faces={surface.face_count:8d} "
            f"area={surface.face_areas.sum():.3f} m2",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "transfer_metrics.csv", index=False)
    pd.DataFrame(cases).to_csv(args.output / "case_summary.csv", index=False)
    summary = _aggregate(metrics)
    summary.to_csv(args.output / "aggregate_metrics.csv", index=False)
    manifest = {
        "schema_version": "1.0",
        "case_count": len(paths),
        "method_count": len(METHODS),
        "target_node_count": len(targets["node_id"]),
        "neighbors": args.neighbors,
        "workers": args.workers,
        "projection_metric": args.projection_metric,
        "duration_s": args.duration_s,
        "time_step_s": args.time_step_s,
        "elapsed_s": time.perf_counter() - started,
        "input_node_sha256": _sha256(args.nodes),
        "input_link_sha256": _sha256(args.links),
        "input_envelope_sha256": _sha256(args.envelope),
        "output_sha256": _sha256(args.output / "transfer_metrics.csv"),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def _load_targets(path: Path, envelope_path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["mass_kg"] >= 0.05].copy().reset_index(drop=True)
    xyz = frame[["centroid_x_m", "centroid_y_m", "centroid_z_m"]].to_numpy(float).copy()
    envelope = np.load(envelope_path)["points_m"].astype(float)
    xyz[:, 0] += 0.70 - float(envelope[:, 0].min())
    xyz[:, 2] += 0.32
    return {
        "node_id": frame["node_id"].astype(str).to_numpy(),
        "xyz": xyz,
        "area": frame["area_m2"].to_numpy(float),
        "capacity": (frame["mass_kg"] * frame["specific_heat_J_kgK"]).to_numpy(float),
        "region": frame["body_region"].astype(str).to_numpy(),
    }


def _load_laplacian(path: Path, node_ids: np.ndarray) -> sparse.csr_matrix:
    frame = pd.read_csv(path)
    lookup = {value: index for index, value in enumerate(node_ids)}
    mapped_i = frame["from_node"].astype(str).map(lookup)
    mapped_j = frame["to_node"].astype(str).map(lookup)
    valid = mapped_i.notna() & mapped_j.notna()
    i = mapped_i.loc[valid].to_numpy(int)
    j = mapped_j.loc[valid].to_numpy(int)
    conductance = frame.loc[valid, "conductance_W_K"].to_numpy(float)
    row = np.concatenate([i, j, i, j])
    col = np.concatenate([j, i, i, j])
    values = np.concatenate([-conductance, -conductance, conductance, conductance])
    return sparse.coo_matrix((values, (row, col)), shape=(len(node_ids), len(node_ids))).tocsr()


def _analyze_case(
    label: str,
    surface: SurfaceField,
    targets: dict[str, np.ndarray],
    laplacian: sparse.csr_matrix,
    *,
    neighbors: int,
    workers: int,
    projection_metric: str,
    duration_s: float,
    time_step_s: float,
) -> tuple[list[dict], dict, dict]:
    valid = np.isfinite(surface.face_areas) & (surface.face_areas > 1.0e-14)
    source_xyz = surface.face_centers[valid]
    source_area = surface.face_areas[valid]
    source_flux = np.asarray(surface.cell_data["wallHeatFlux"])[valid]
    source_temperature = np.asarray(surface.cell_data.get("T", np.full(surface.face_count, np.nan)))[valid]
    target_xyz = targets["xyz"]
    target_area = targets["area"]
    capacity = targets["capacity"]
    source_scaled, target_scaled = scaled_coordinates(source_xyz, target_xyz)

    target_tree = cKDTree(target_scaled)
    count = min(neighbors, len(target_xyz))
    source_distance, source_index = target_tree.query(source_scaled, k=count, workers=workers)
    if count == 1:
        source_distance = source_distance[:, None]
        source_index = source_index[:, None]
    positive = source_distance[source_distance > 0.0]
    bandwidth = max(float(np.median(positive)), 1.0e-12)
    local_weights = np.exp(-0.5 * (source_distance / bandwidth) ** 2)
    local_weights /= np.maximum(local_weights.sum(axis=1, keepdims=True), 1.0e-30)
    source_rate = source_flux * source_area

    source_tree = cKDTree(source_scaled)
    target_distance, target_source_index = source_tree.query(
        target_scaled,
        k=min(8, len(source_scaled)),
        workers=workers,
    )
    if target_source_index.ndim == 1:
        target_source_index = target_source_index[:, None]
        target_distance = target_distance[:, None]
    inverse = 1.0 / np.maximum(target_distance, 1.0e-10) ** 2
    inverse /= inverse.sum(axis=1, keepdims=True)

    control_area = np.bincount(source_index[:, 0], weights=source_area, minlength=len(target_xyz))
    nearest_flux_rate = source_flux[target_source_index[:, 0]] * control_area
    idw_flux_rate = np.sum(source_flux[target_source_index] * inverse, axis=1) * control_area
    local_rate = np.bincount(
        source_index.ravel(),
        weights=(local_weights * source_rate[:, None]).ravel(),
        minlength=len(target_xyz),
    )
    local_support_area = np.bincount(
        source_index.ravel(),
        weights=(local_weights * source_area[:, None]).ravel(),
        minlength=len(target_xyz),
    )
    voronoi_rate = np.bincount(source_index[:, 0], weights=source_rate, minlength=len(target_xyz))
    metric_weights = {
        "capacity": capacity,
        "uniform": np.ones_like(capacity),
        "surface_area": local_support_area,
    }[projection_metric]
    projected_rate, projection = _moment_projection(
        local_rate,
        source_rate,
        source_xyz,
        target_xyz,
        metric_weights * (control_area > 0.0),
    )
    mappings = {
        "nearest_flux": nearest_flux_rate,
        "inverse_distance_flux": idw_flux_rate,
        "conservative_local_rate": local_rate,
        "conservative_moment_projection": projected_rate,
    }
    support_areas = {
        "nearest_flux": control_area,
        "inverse_distance_flux": control_area,
        "conservative_local_rate": local_support_area,
        "conservative_moment_projection": local_support_area,
    }

    source_moments, target_moments = moment_matrices(source_xyz, target_xyz)
    reference_constraints = source_moments @ source_rate
    reference_temperature = _thermal_response(
        local_rate,
        capacity,
        control_area,
        laplacian,
        duration_s,
        time_step_s,
    )
    rows = []
    source_l1 = max(float(np.sum(np.abs(source_rate))), 1.0e-30)
    source_flux_rms = max(float(np.sqrt(np.sum(source_area * source_flux**2) / np.sum(source_area))), 1.0e-30)
    voronoi_rate_rms = max(float(np.sqrt(np.mean(voronoi_rate**2))), 1.0e-30)
    local_rate_rms = max(float(np.sqrt(np.mean(local_rate**2))), 1.0e-30)
    for method, target_rate in mappings.items():
        constraints = target_moments @ target_rate
        support_area = support_areas[method]
        target_flux = target_rate / np.maximum(support_area, 1.0e-14)
        reconstructed_flux = np.sum(target_flux[source_index] * local_weights, axis=1)
        flux_rmse = float(np.sqrt(np.sum(source_area * (reconstructed_flux - source_flux) ** 2) / np.sum(source_area)))
        response = _thermal_response(
            target_rate,
            capacity,
            control_area,
            laplacian,
            duration_s,
            time_step_s,
        )
        graph_energy = max(float(target_rate @ (laplacian @ target_rate)), 0.0)
        rows.append(
            {
                "case": label,
                "method": method,
                "source_faces": len(source_xyz),
                "source_total_rate_W": reference_constraints[0],
                "mapped_total_rate_W": constraints[0],
                "relative_heat_error": abs(float(constraints[0] - reference_constraints[0])) / source_l1,
                "relative_first_moment_error": float(
                    np.linalg.norm(constraints[1:] - reference_constraints[1:]) / source_l1
                ),
                "relative_voronoi_rate_rmse": float(
                    np.sqrt(np.mean((target_rate - voronoi_rate) ** 2)) / voronoi_rate_rms
                ),
                "relative_rate_change_from_local": float(
                    np.sqrt(np.mean((target_rate - local_rate) ** 2)) / local_rate_rms
                ),
                "relative_back_projection_rmse": flux_rmse / source_flux_rms,
                "final_temperature_change_C": float(np.sqrt(np.mean((response - reference_temperature) ** 2))),
                "relative_final_temperature_change": float(
                    np.sqrt(np.mean((response - reference_temperature) ** 2))
                    / max(np.sqrt(np.mean(reference_temperature**2)), 1.0e-30)
                ),
                "capacity_mean_temperature_change_C": abs(
                    float(np.sum(capacity * (response - reference_temperature)) / np.sum(capacity))
                ),
                "negative_rate_fraction": float(np.mean(target_rate[support_area > 0.0] < 0.0)),
                "target_rate_l2_W": float(np.linalg.norm(target_rate)),
                "normalized_graph_roughness": float(
                    np.sqrt(graph_energy / max(float(target_rate @ target_rate), 1.0e-30))
                ),
            }
        )

    sample = np.linspace(0, len(source_xyz) - 1, min(80_000, len(source_xyz)), dtype=int)
    representative = {
        "case": np.array(label),
        "source_xyz": source_xyz[sample].astype(np.float32),
        "source_flux_W_m2": source_flux[sample].astype(np.float32),
        "source_temperature_K": source_temperature[sample].astype(np.float32),
        "target_xyz": target_xyz.astype(np.float32),
        "target_area_m2": target_area.astype(np.float32),
        "target_control_area_m2": control_area.astype(np.float32),
        "target_local_support_area_m2": local_support_area.astype(np.float32),
        "voronoi_rate_W": voronoi_rate.astype(np.float32),
        **{f"{method}_rate_W": values.astype(np.float32) for method, values in mappings.items()},
    }
    case_summary = {
        "case": label,
        "point_count": surface.point_count,
        "face_count": surface.face_count,
        "valid_face_count": len(source_xyz),
        "surface_area_m2": float(source_area.sum()),
        "active_target_node_count": int(np.count_nonzero(control_area > 0.0)),
        "wall_heat_flux_min_W_m2": float(source_flux.min()),
        "wall_heat_flux_mean_W_m2": float(np.average(source_flux, weights=source_area)),
        "wall_heat_flux_max_W_m2": float(source_flux.max()),
        "wall_temperature_min_K": float(np.nanmin(source_temperature)),
        "wall_temperature_mean_K": float(np.average(source_temperature, weights=source_area)),
        "wall_temperature_max_K": float(np.nanmax(source_temperature)),
        "projection_condition_number": projection["condition_number"],
        "projection_metric": projection_metric,
        "projection_relative_correction": projection["relative_correction"],
        "projection_constraint_residual": projection["constraint_residual"],
    }
    return rows, case_summary, representative


def _moment_projection(
    local_rate: np.ndarray,
    source_rate: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source_moments, target_moments = moment_matrices(source_xyz, target_xyz)
    scaled_capacity = capacity / np.mean(capacity)
    gram = (target_moments * scaled_capacity[None, :]) @ target_moments.T
    mismatch = source_moments @ source_rate - target_moments @ local_rate
    correction = scaled_capacity * (target_moments.T @ np.linalg.solve(gram, mismatch))
    projected = local_rate + correction
    residual = source_moments @ source_rate - target_moments @ projected
    return projected, {
        "condition_number": float(np.linalg.cond(gram)),
        "relative_correction": float(np.linalg.norm(correction) / max(np.linalg.norm(local_rate), 1.0e-30)),
        "constraint_residual": float(np.linalg.norm(residual) / max(np.sum(np.abs(source_rate)), 1.0e-30)),
    }


def _thermal_response(
    heat_rate: np.ndarray,
    capacity: np.ndarray,
    area: np.ndarray,
    laplacian: sparse.csr_matrix,
    duration_s: float,
    time_step_s: float,
) -> np.ndarray:
    sink = 2.5 * area
    matrix = sparse.diags(capacity / time_step_s + sink) + laplacian
    solve = factorized(matrix.tocsc())
    temperature = np.zeros_like(capacity)
    steps = int(np.ceil(duration_s / time_step_s))
    for _ in range(steps):
        temperature = solve(heat_rate + capacity * temperature / time_step_s)
    return temperature


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "relative_heat_error",
        "relative_first_moment_error",
        "relative_voronoi_rate_rmse",
        "relative_rate_change_from_local",
        "relative_back_projection_rmse",
        "final_temperature_change_C",
        "relative_final_temperature_change",
        "capacity_mean_temperature_change_C",
        "negative_rate_fraction",
        "normalized_graph_roughness",
    ]
    rows = []
    for method, group in metrics.groupby("method", sort=False):
        row: dict[str, float | str | int] = {"method": method, "case_count": len(group)}
        for column in numeric:
            values = group[column].to_numpy(float)
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _case_label(path: Path, root: Path) -> str:
    return path.relative_to(root).parts[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
