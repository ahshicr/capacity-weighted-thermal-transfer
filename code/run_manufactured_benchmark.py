from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree


Array = np.ndarray
Field = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class QuadMesh:
    nx: int
    ny: int
    nodes: Array
    vertices: Array
    centers: Array
    areas: Array


@dataclass(frozen=True)
class ThermalModel:
    capacity: Array
    solve_step: Callable[[Array], Array]
    capacity_over_dt: Array
    steps: int
    dt: float


def polygon_area_centroid(vertices: Array) -> tuple[Array, Array]:
    shifted = np.roll(vertices, -1, axis=1)
    cross = vertices[:, :, 0] * shifted[:, :, 1] - shifted[:, :, 0] * vertices[:, :, 1]
    twice_area = cross.sum(axis=1)
    if np.any(twice_area <= 1.0e-14):
        raise ValueError("mesh contains an inverted or degenerate cell")
    area = 0.5 * twice_area
    center_x = ((vertices[:, :, 0] + shifted[:, :, 0]) * cross).sum(axis=1) / (3.0 * twice_area)
    center_y = ((vertices[:, :, 1] + shifted[:, :, 1]) * cross).sum(axis=1) / (3.0 * twice_area)
    return area, np.column_stack([center_x, center_y])


def make_mesh(nx: int, ny: int, warp: float) -> QuadMesh:
    u = np.linspace(0.0, 1.0, nx + 1)
    v = np.linspace(0.0, 1.0, ny + 1)
    uu, vv = np.meshgrid(u, v)
    xx = uu + warp * np.sin(np.pi * uu) * np.sin(2.0 * np.pi * vv)
    yy = vv + warp * np.sin(np.pi * vv) * np.sin(2.0 * np.pi * uu)
    nodes = np.stack([xx, yy], axis=-1)
    vertices = np.stack(
        [
            nodes[:-1, :-1],
            nodes[:-1, 1:],
            nodes[1:, 1:],
            nodes[1:, :-1],
        ],
        axis=2,
    ).reshape(-1, 4, 2)
    areas, centers = polygon_area_centroid(vertices)
    if not np.isclose(areas.sum(), 1.0, rtol=0.0, atol=2.0e-12):
        raise ValueError("mesh does not cover the unit square")
    return QuadMesh(nx=nx, ny=ny, nodes=nodes, vertices=vertices, centers=centers, areas=areas)


def field_linear(x: Array, y: Array) -> Array:
    return 720.0 + 310.0 * x + 190.0 * y


def field_smooth_hotspot(x: Array, y: Array) -> Array:
    hotspot = 510.0 * np.exp(-((x - 0.72) ** 2 / 0.012 + (y - 0.34) ** 2 / 0.025))
    wave = 145.0 * np.sin(2.0 * np.pi * x) * np.cos(np.pi * y)
    return 760.0 + 185.0 * x + 95.0 * y + wave + hotspot


def field_thermal_front(x: Array, y: Array) -> Array:
    front = 620.0 / (1.0 + np.exp(-34.0 * (x + 0.42 * y - 0.66)))
    spot = 280.0 * np.exp(-((x - 0.24) ** 2 + (y - 0.76) ** 2) / 0.018)
    return 530.0 + front + spot


def field_multiscale(x: Array, y: Array) -> Array:
    fine = 155.0 * np.sin(6.0 * np.pi * x + 0.25) * np.cos(4.0 * np.pi * y)
    ridge = 390.0 * np.exp(-((y - 0.18 - 0.10 * np.sin(2.0 * np.pi * x)) ** 2) / 0.006)
    return 880.0 + 80.0 * x + fine + ridge


FIELDS: dict[str, Field] = {
    "linear": field_linear,
    "smooth_hotspot": field_smooth_hotspot,
    "thermal_front": field_thermal_front,
    "multiscale": field_multiscale,
}


def integrate_cells(mesh: QuadMesh, field: Field, order: int = 6) -> tuple[Array, Array, Array]:
    points, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    p00 = mesh.vertices[:, 0]
    p10 = mesh.vertices[:, 1]
    p11 = mesh.vertices[:, 2]
    p01 = mesh.vertices[:, 3]
    load = np.zeros(mesh.areas.size)
    x_moment = np.zeros(mesh.areas.size)
    y_moment = np.zeros(mesh.areas.size)
    for i, r in enumerate(points):
        for j, s in enumerate(points):
            mapped = (
                (1.0 - r) * (1.0 - s) * p00
                + r * (1.0 - s) * p10
                + r * s * p11
                + (1.0 - r) * s * p01
            )
            dr = -(1.0 - s) * p00 + (1.0 - s) * p10 + s * p11 - s * p01
            ds = -(1.0 - r) * p00 - r * p10 + r * p11 + (1.0 - r) * p01
            jacobian = np.abs(dr[:, 0] * ds[:, 1] - dr[:, 1] * ds[:, 0])
            flux = field(mapped[:, 0], mapped[:, 1])
            factor = weights[i] * weights[j] * jacobian
            load += factor * flux
            x_moment += factor * flux * mapped[:, 0]
            y_moment += factor * flux * mapped[:, 1]
    return load, x_moment, y_moment


def scaled_moment_matrices(source_xy: Array, target_xy: Array) -> tuple[Array, Array]:
    all_xy = np.vstack([source_xy, target_xy])
    center = 0.5 * (all_xy.min(axis=0) + all_xy.max(axis=0))
    scale = np.maximum(0.5 * (all_xy.max(axis=0) - all_xy.min(axis=0)), 1.0e-12)
    source_scaled = (source_xy - center) / scale
    target_scaled = (target_xy - center) / scale
    return np.vstack([np.ones(source_xy.shape[0]), source_scaled.T]), np.vstack(
        [np.ones(target_xy.shape[0]), target_scaled.T]
    )


def build_local_distribution(source_xy: Array, target_xy: Array, neighbors: int) -> tuple[Array, Array, float]:
    started = time.perf_counter()
    count = min(neighbors, target_xy.shape[0])
    distances, indices = cKDTree(target_xy).query(source_xy, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    positive = distances[distances > 0.0]
    bandwidth = max(float(np.median(positive)) if positive.size else 1.0, 1.0e-12)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-30)
    return indices, weights, time.perf_counter() - started


def apply_local_distribution(indices: Array, weights: Array, source_load: Array, target_count: int) -> Array:
    return np.bincount(
        indices.ravel(),
        weights=(weights * source_load[:, None]).ravel(),
        minlength=target_count,
    )


def project_moments(
    provisional: Array,
    source_load: Array,
    source_moments: Array,
    target_moments: Array,
    metric: Array,
) -> tuple[Array, float, float]:
    metric = np.asarray(metric, dtype=float)
    gram = (target_moments * metric[None, :]) @ target_moments.T
    condition = float(np.linalg.cond(gram))
    residual = source_moments @ source_load - target_moments @ provisional
    correction = metric * (target_moments.T @ np.linalg.solve(gram, residual))
    projected = provisional + correction
    relative_correction = float(np.linalg.norm(correction) / max(np.linalg.norm(provisional), 1.0e-30))
    return projected, condition, relative_correction


def clip_polygon(subject: list[Array], clipper: Array) -> list[Array]:
    output = subject
    for edge_index in range(clipper.shape[0]):
        edge_start = clipper[edge_index]
        edge_end = clipper[(edge_index + 1) % clipper.shape[0]]
        edge = edge_end - edge_start

        def inside(point: Array) -> bool:
            relative = point - edge_start
            return edge[0] * relative[1] - edge[1] * relative[0] >= -1.0e-13

        def intersection(first: Array, second: Array) -> Array:
            direction = second - first
            denominator = edge[0] * direction[1] - edge[1] * direction[0]
            if abs(denominator) < 1.0e-15:
                return 0.5 * (first + second)
            relative = edge_start - first
            parameter = (edge[0] * relative[1] - edge[1] * relative[0]) / denominator
            return first + parameter * direction

        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        previous_inside = inside(previous)
        for current in input_polygon:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
    return output


def polygon_area(points: list[Array]) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points)
    shifted = np.roll(array, -1, axis=0)
    return 0.5 * abs(float(np.sum(array[:, 0] * shifted[:, 1] - shifted[:, 0] * array[:, 1])))


def build_common_refinement(source: QuadMesh, target: QuadMesh) -> tuple[Array, Array, Array, float, float]:
    started = time.perf_counter()
    rows: list[int] = []
    columns: list[int] = []
    overlaps: list[float] = []
    covered = np.zeros(target.areas.size)
    for target_index, target_polygon in enumerate(target.vertices):
        minimum = target_polygon.min(axis=0)
        maximum = target_polygon.max(axis=0)
        i_min = max(0, int(np.floor(minimum[0] * source.nx - 1.0e-12)))
        i_max = min(source.nx - 1, int(np.floor(maximum[0] * source.nx + 1.0e-12)))
        j_min = max(0, int(np.floor(minimum[1] * source.ny - 1.0e-12)))
        j_max = min(source.ny - 1, int(np.floor(maximum[1] * source.ny + 1.0e-12)))
        for j in range(j_min, j_max + 1):
            y0 = j / source.ny
            y1 = (j + 1) / source.ny
            for i in range(i_min, i_max + 1):
                x0 = i / source.nx
                x1 = (i + 1) / source.nx
                rectangle = [
                    np.array([x0, y0]),
                    np.array([x1, y0]),
                    np.array([x1, y1]),
                    np.array([x0, y1]),
                ]
                area = polygon_area(clip_polygon(rectangle, target_polygon))
                if area > 1.0e-14:
                    rows.append(target_index)
                    columns.append(j * source.nx + i)
                    overlaps.append(area)
                    covered[target_index] += area
    relative_coverage_error = float(
        np.max(np.abs(covered - target.areas) / np.maximum(target.areas, 1.0e-30))
    )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(overlaps, dtype=float),
        time.perf_counter() - started,
        relative_coverage_error,
    )


def apply_common_refinement(
    rows: Array,
    columns: Array,
    overlaps: Array,
    source_flux: Array,
    target_count: int,
) -> Array:
    return np.bincount(rows, weights=overlaps * source_flux[columns], minlength=target_count)


def make_thermal_model(mesh: QuadMesh, duration: float = 600.0, dt: float = 10.0) -> ThermalModel:
    x = mesh.centers[:, 0]
    y = mesh.centers[:, 1]
    material_factor = 0.45 + 2.7 * x + 0.9 * y + 1.3 * np.exp(-((x - 0.72) ** 2 + (y - 0.34) ** 2) / 0.08)
    capacity = 4.2e4 * mesh.areas * material_factor
    row: list[int] = []
    col: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(mesh.areas.size)
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            current = j * mesh.nx + i
            for ni, nj in ((i + 1, j), (i, j + 1)):
                if ni >= mesh.nx or nj >= mesh.ny:
                    continue
                neighbor = nj * mesh.nx + ni
                distance = np.linalg.norm(mesh.centers[current] - mesh.centers[neighbor])
                length = 0.5 * (np.sqrt(mesh.areas[current]) + np.sqrt(mesh.areas[neighbor]))
                conductance = 3.5 * length / max(distance, 1.0e-12)
                row.extend([current, neighbor])
                col.extend([neighbor, current])
                values.extend([-conductance, -conductance])
                diagonal[current] += conductance
                diagonal[neighbor] += conductance
    convection = 1.8 * mesh.areas
    row.extend(range(mesh.areas.size))
    col.extend(range(mesh.areas.size))
    values.extend((diagonal + convection).tolist())
    operator = coo_matrix((values, (row, col)), shape=(mesh.areas.size, mesh.areas.size)).tocsc()
    capacity_over_dt = capacity / dt
    matrix = operator + diags(capacity_over_dt, offsets=0, format="csc")
    return ThermalModel(
        capacity=capacity,
        solve_step=factorized(matrix),
        capacity_over_dt=capacity_over_dt,
        steps=int(round(duration / dt)),
        dt=dt,
    )


def thermal_trajectory(load: Array, model: ThermalModel) -> Array:
    temperature = np.zeros(load.size)
    trajectory = np.empty((model.steps + 1, load.size), dtype=float)
    trajectory[0] = temperature
    for step in range(1, model.steps + 1):
        temperature = model.solve_step(model.capacity_over_dt * temperature + load)
        trajectory[step] = temperature
    return trajectory


def relative_norm(error: Array, reference: Array) -> float:
    return float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1.0e-30))


def compute_metrics(
    predicted: Array,
    exact: Array,
    source_load: Array,
    source_moments: Array,
    target_moments: Array,
    exact_trajectory: Array,
    predicted_trajectory: Array,
) -> dict[str, float]:
    exact_constraints = target_moments @ exact
    predicted_constraints = target_moments @ predicted
    source_constraints = source_moments @ source_load
    temperature_error = predicted_trajectory - exact_trajectory
    temperature_scale = max(float(np.sqrt(np.mean(exact_trajectory**2))), 1.0e-30)
    return {
        "load_l2_relative": relative_norm(predicted - exact, exact),
        "total_error_exact": abs(float(predicted.sum() - exact.sum())) / max(abs(float(exact.sum())), 1.0e-30),
        "moment_error_exact": float(
            np.linalg.norm(predicted_constraints[1:] - exact_constraints[1:])
            / max(np.linalg.norm(exact_constraints[1:]), abs(float(exact_constraints[0])), 1.0e-30)
        ),
        "source_total_residual": abs(float(predicted_constraints[0] - source_constraints[0]))
        / max(abs(float(source_constraints[0])), 1.0e-30),
        "source_moment_residual": float(
            np.linalg.norm(predicted_constraints[1:] - source_constraints[1:])
            / max(np.linalg.norm(source_constraints[1:]), abs(float(source_constraints[0])), 1.0e-30)
        ),
        "thermal_rmse_relative": float(np.sqrt(np.mean(temperature_error**2)) / temperature_scale),
        "thermal_max_absolute": float(np.max(np.abs(temperature_error))),
        "negative_load_fraction": float(np.mean(predicted < 0.0)),
    }


def method_predictions(
    source: QuadMesh,
    target: QuadMesh,
    source_load: Array,
    source_flux: Array,
    source_moments: Array,
    target_moments: Array,
    local_indices: Array,
    local_weights: Array,
    common_rows: Array,
    common_columns: Array,
    common_overlaps: Array,
    capacity_metric: Array,
) -> tuple[dict[str, Array], dict[str, dict[str, float]]]:
    predictions: dict[str, Array] = {}
    timing: dict[str, dict[str, float]] = {}
    source_tree = cKDTree(source.centers)

    started = time.perf_counter()
    _, nearest_index = source_tree.query(target.centers, k=1)
    predictions["nearest"] = source_flux[nearest_index] * target.areas
    timing["nearest"] = {"setup_s": 0.0, "apply_s": time.perf_counter() - started}

    started = time.perf_counter()
    count = min(8, source.centers.shape[0])
    distances, indices = source_tree.query(target.centers, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    inverse = 1.0 / np.maximum(distances, 1.0e-12) ** 2
    inverse /= inverse.sum(axis=1, keepdims=True)
    interpolated_flux = np.sum(inverse * source_flux[indices], axis=1)
    predictions["inverse_distance"] = interpolated_flux * target.areas
    timing["inverse_distance"] = {"setup_s": 0.0, "apply_s": time.perf_counter() - started}

    started = time.perf_counter()
    rbf = RBFInterpolator(
        source.centers,
        source_flux,
        neighbors=min(32, source.centers.shape[0]),
        smoothing=1.0e-10,
        kernel="thin_plate_spline",
        degree=1,
    )
    setup_rbf = time.perf_counter() - started
    started = time.perf_counter()
    rbf_load = np.asarray(rbf(target.centers), dtype=float) * target.areas
    predictions["local_rbf"] = rbf_load
    timing["local_rbf"] = {"setup_s": setup_rbf, "apply_s": time.perf_counter() - started}

    started = time.perf_counter()
    scaled_rbf = rbf_load * (source_load.sum() / max(rbf_load.sum(), 1.0e-30))
    predictions["total_scaled_rbf"] = scaled_rbf
    timing["total_scaled_rbf"] = {"setup_s": setup_rbf, "apply_s": time.perf_counter() - started}

    started = time.perf_counter()
    rbf_uniform_projected, rbf_uniform_condition, rbf_uniform_correction = project_moments(
        rbf_load, source_load, source_moments, target_moments, np.ones(target.areas.size)
    )
    predictions["rbf_uniform_moment"] = rbf_uniform_projected
    timing["rbf_uniform_moment"] = {
        "setup_s": setup_rbf,
        "apply_s": time.perf_counter() - started,
        "condition_number": rbf_uniform_condition,
        "relative_correction": rbf_uniform_correction,
    }

    started = time.perf_counter()
    rbf_capacity_projected, rbf_capacity_condition, rbf_capacity_correction = project_moments(
        rbf_load, source_load, source_moments, target_moments, capacity_metric
    )
    predictions["rbf_capacity_moment"] = rbf_capacity_projected
    timing["rbf_capacity_moment"] = {
        "setup_s": setup_rbf,
        "apply_s": time.perf_counter() - started,
        "condition_number": rbf_capacity_condition,
        "relative_correction": rbf_capacity_correction,
    }

    started = time.perf_counter()
    local = apply_local_distribution(local_indices, local_weights, source_load, target.areas.size)
    predictions["local_conservative"] = local
    timing["local_conservative"] = {"setup_s": 0.0, "apply_s": time.perf_counter() - started}

    started = time.perf_counter()
    uniform_projected, uniform_condition, uniform_correction = project_moments(
        local, source_load, source_moments, target_moments, np.ones(target.areas.size)
    )
    predictions["uniform_moment"] = uniform_projected
    timing["uniform_moment"] = {
        "setup_s": 0.0,
        "apply_s": time.perf_counter() - started,
        "condition_number": uniform_condition,
        "relative_correction": uniform_correction,
    }

    started = time.perf_counter()
    capacity_projected, capacity_condition, capacity_correction = project_moments(
        local, source_load, source_moments, target_moments, capacity_metric
    )
    predictions["capacity_moment"] = capacity_projected
    timing["capacity_moment"] = {
        "setup_s": 0.0,
        "apply_s": time.perf_counter() - started,
        "condition_number": capacity_condition,
        "relative_correction": capacity_correction,
    }

    started = time.perf_counter()
    predictions["common_refinement"] = apply_common_refinement(
        common_rows, common_columns, common_overlaps, source_flux, target.areas.size
    )
    timing["common_refinement"] = {"setup_s": 0.0, "apply_s": time.perf_counter() - started}
    return predictions, timing


def run_accuracy_benchmark(output_dir: Path) -> tuple[list[dict[str, object]], dict[str, Array]]:
    source_levels = [(16, 8), (32, 16), (64, 32), (128, 64)]
    target_levels = [(14, 7), (28, 14), (56, 28)]
    records: list[dict[str, object]] = []
    representative: dict[str, Array] = {}
    for source_nx, source_ny in source_levels:
        source = make_mesh(source_nx, source_ny, warp=0.0)
        for target_nx, target_ny in target_levels:
            target = make_mesh(target_nx, target_ny, warp=0.035)
            source_moments, target_moments = scaled_moment_matrices(source.centers, target.centers)
            local_indices, local_weights, local_setup = build_local_distribution(
                source.centers, target.centers, neighbors=8
            )
            common_rows, common_columns, common_overlaps, common_setup, coverage_error = build_common_refinement(
                source, target
            )
            if coverage_error > 2.0e-10:
                raise RuntimeError(f"common-refinement coverage failed: {coverage_error:.3e}")
            thermal_model = make_thermal_model(target)
            for field_name, field in FIELDS.items():
                source_load, _, _ = integrate_cells(source, field)
                exact_target_load, _, _ = integrate_cells(target, field)
                source_flux = source_load / source.areas
                predictions, timing = method_predictions(
                    source,
                    target,
                    source_load,
                    source_flux,
                    source_moments,
                    target_moments,
                    local_indices,
                    local_weights,
                    common_rows,
                    common_columns,
                    common_overlaps,
                    thermal_model.capacity,
                )
                exact_trajectory = thermal_trajectory(exact_target_load, thermal_model)
                trajectories = {
                    method: thermal_trajectory(predicted, thermal_model)
                    for method, predicted in predictions.items()
                }
                rbf_load = predictions["local_rbf"]
                rbf_trajectory = trajectories["local_rbf"]
                rbf_energy_scale = max(
                    float(np.sqrt(np.sum(rbf_load**2 / thermal_model.capacity))), 1.0e-30
                )
                rbf_temperature_scale = max(float(np.sqrt(np.mean(rbf_trajectory**2))), 1.0e-30)
                for method, predicted in predictions.items():
                    predicted_trajectory = trajectories[method]
                    metrics = compute_metrics(
                        predicted,
                        exact_target_load,
                        source_load,
                        source_moments,
                        target_moments,
                        exact_trajectory,
                        predicted_trajectory,
                    )
                    correction = predicted - rbf_load
                    metrics["rbf_correction_energy_relative"] = float(
                        np.sqrt(np.sum(correction**2 / thermal_model.capacity)) / rbf_energy_scale
                    )
                    metrics["rbf_thermal_deviation_relative"] = float(
                        np.sqrt(np.mean((predicted_trajectory - rbf_trajectory) ** 2))
                        / rbf_temperature_scale
                    )
                    details = timing[method]
                    records.append(
                        {
                            "source_nx": source_nx,
                            "source_ny": source_ny,
                            "source_cells": source.areas.size,
                            "target_nx": target_nx,
                            "target_ny": target_ny,
                            "target_cells": target.areas.size,
                            "warp_amplitude": 0.035,
                            "field": field_name,
                            "method": method,
                            **metrics,
                            "setup_s": float(details.get("setup_s", 0.0)),
                            "apply_s": float(details.get("apply_s", 0.0)),
                            "shared_local_setup_s": local_setup,
                            "shared_common_setup_s": common_setup,
                            "condition_number": float(details.get("condition_number", np.nan)),
                            "relative_correction": float(details.get("relative_correction", np.nan)),
                            "coverage_error": coverage_error,
                        }
                    )
                if source_nx == 64 and target_nx == 56 and field_name == "thermal_front":
                    representative = {
                        "source_vertices": source.vertices,
                        "source_centers": source.centers,
                        "source_load": source_load,
                        "target_vertices": target.vertices,
                        "target_centers": target.centers,
                        "target_areas": target.areas,
                        "exact_load": exact_target_load,
                        "capacity": thermal_model.capacity,
                        "time": np.arange(thermal_model.steps + 1) * thermal_model.dt,
                        "exact_temperature": exact_trajectory,
                    }
                    for method, predicted in predictions.items():
                        representative[f"load_{method}"] = predicted
                        representative[f"temperature_{method}"] = thermal_trajectory(predicted, thermal_model)
    if not representative:
        raise RuntimeError("representative case was not captured")
    return records, representative


def run_scaling_benchmark() -> list[dict[str, object]]:
    generator = np.random.default_rng(20260717)
    target_count = 421
    target_xy = generator.random((target_count, 2))
    target_capacity = 0.5 + 2.5 * target_xy[:, 0] + 0.7 * target_xy[:, 1]
    records: list[dict[str, object]] = []
    for source_count in (4096, 16384, 65536, 262144, 1048576):
        source_xy = generator.random((source_count, 2))
        source_load = 0.2 + generator.random(source_count)
        source_moments, target_moments = scaled_moment_matrices(source_xy, target_xy)
        elapsed: list[float] = []
        total_residual: list[float] = []
        moment_residual: list[float] = []
        correction: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            indices, weights, _ = build_local_distribution(source_xy, target_xy, neighbors=8)
            provisional = apply_local_distribution(indices, weights, source_load, target_count)
            projected, _, relative_correction = project_moments(
                provisional, source_load, source_moments, target_moments, target_capacity
            )
            elapsed.append(time.perf_counter() - started)
            source_constraints = source_moments @ source_load
            target_constraints = target_moments @ projected
            total_residual.append(
                abs(float(target_constraints[0] - source_constraints[0]))
                / max(abs(float(source_constraints[0])), 1.0e-30)
            )
            moment_residual.append(
                float(
                    np.linalg.norm(target_constraints[1:] - source_constraints[1:])
                    / max(np.linalg.norm(source_constraints[1:]), abs(float(source_constraints[0])), 1.0e-30)
                )
            )
            correction.append(relative_correction)
        matrix_free_bytes = source_count * (2 * 8 + 8 * (8 + 8 + 8)) + target_count * (2 * 8 + 8)
        dense_bytes = source_count * target_count * 8
        records.append(
            {
                "source_cells": source_count,
                "target_nodes": target_count,
                "neighbors": 8,
                "repeats": 3,
                "median_runtime_s": float(np.median(elapsed)),
                "minimum_runtime_s": float(np.min(elapsed)),
                "maximum_runtime_s": float(np.max(elapsed)),
                "estimated_matrix_free_memory_mb": matrix_free_bytes / 1024.0**2,
                "dense_operator_memory_mb": dense_bytes / 1024.0**2,
                "maximum_total_residual": float(np.max(total_residual)),
                "maximum_moment_residual": float(np.max(moment_residual)),
                "median_relative_correction": float(np.median(correction)),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_accuracy(records: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = sorted({str(record["method"]) for record in records})
    metrics = [
        "load_l2_relative",
        "total_error_exact",
        "moment_error_exact",
        "source_total_residual",
        "source_moment_residual",
        "thermal_rmse_relative",
        "thermal_max_absolute",
        "negative_load_fraction",
        "rbf_correction_energy_relative",
        "rbf_thermal_deviation_relative",
        "apply_s",
    ]
    aggregate: list[dict[str, object]] = []
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        row: dict[str, object] = {"method": method, "cases": len(selected)}
        for metric in metrics:
            values = np.asarray([float(record[metric]) for record in selected])
            row[f"median_{metric}"] = float(np.median(values))
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"maximum_{metric}"] = float(np.max(values))
        aggregate.append(row)
    return aggregate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quadrature_audit() -> dict[str, object]:
    audits: list[dict[str, object]] = []
    for mesh_name, mesh in (
        ("coarse_source", make_mesh(16, 8, warp=0.0)),
        ("coarse_target", make_mesh(14, 7, warp=0.035)),
    ):
        for field_name, field in FIELDS.items():
            order_six, _, _ = integrate_cells(mesh, field, order=6)
            order_ten, _, _ = integrate_cells(mesh, field, order=10)
            audits.append(
                {
                    "mesh": mesh_name,
                    "field": field_name,
                    "relative_difference": relative_norm(order_six - order_ten, order_ten),
                    "total_difference": abs(float(order_six.sum() - order_ten.sum()))
                    / max(abs(float(order_ten.sum())), 1.0e-30),
                }
            )
    return {
        "comparisons": audits,
        "maximum_relative_difference": max(float(item["relative_difference"]) for item in audits),
        "maximum_total_difference": max(float(item["total_difference"]) for item in audits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nonmatching-mesh manufactured benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "manufactured_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    accuracy_records, representative = run_accuracy_benchmark(output_dir)
    scaling_records = run_scaling_benchmark()
    integration_audit = quadrature_audit()
    accuracy_path = output_dir / "accuracy_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    scaling_path = output_dir / "runtime_scaling.csv"
    representative_path = output_dir / "representative_case.npz"
    write_csv(accuracy_path, accuracy_records)
    write_csv(aggregate_path, aggregate_accuracy(accuracy_records))
    write_csv(scaling_path, scaling_records)
    np.savez_compressed(representative_path, **representative)
    manifest = {
        "benchmark": "nonmatching quadrilateral manufactured solution",
        "full_accuracy_cases": len(accuracy_records),
        "unique_physical_scenarios": len(accuracy_records) // len({str(record["method"]) for record in accuracy_records}),
        "methods": sorted({str(record["method"]) for record in accuracy_records}),
        "fields": list(FIELDS),
        "source_levels": [[16, 8], [32, 16], [64, 32], [128, 64]],
        "target_levels": [[14, 7], [28, 14], [56, 28]],
        "quadrature_order": 6,
        "quadrature_audit": integration_audit,
        "warp_amplitude": 0.035,
        "elapsed_s": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "outputs": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in (accuracy_path, aggregate_path, scaling_path, representative_path)
        },
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
