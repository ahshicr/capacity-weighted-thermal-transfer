from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TransferDiagnostics:
    heat_rate_error: float
    moment_error: float
    correction_norm: float
    condition_number: float


def scaled_coordinates(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale both point clouds with one affine map."""
    all_xyz = np.vstack([source_xyz, target_xyz]).astype(float)
    center = 0.5 * (all_xyz.min(axis=0) + all_xyz.max(axis=0))
    scale = np.maximum(0.5 * (all_xyz.max(axis=0) - all_xyz.min(axis=0)), 1.0e-12)
    return (source_xyz - center) / scale, (target_xyz - center) / scale


def local_kernel(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    neighbors: int = 16,
    length_scale: float | None = None,
) -> np.ndarray:
    """Return a nonnegative column-stochastic local transfer matrix."""
    if source_xyz.ndim != 2 or target_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must have shape (count, 3)")
    if target_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must have shape (count, 3)")
    count = min(int(neighbors), target_xyz.shape[0])
    if count < 1:
        raise ValueError("at least one target point is required")

    source_scaled, target_scaled = scaled_coordinates(source_xyz, target_xyz)
    distances, indices = cKDTree(target_scaled).query(source_scaled, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    positive = distances[distances > 0.0]
    bandwidth = float(np.median(positive)) if length_scale is None and positive.size else 1.0
    if length_scale is not None:
        bandwidth = float(length_scale)
    bandwidth = max(bandwidth, 1.0e-12)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-30)

    kernel = np.zeros((target_xyz.shape[0], source_xyz.shape[0]), dtype=float)
    source_columns = np.arange(source_xyz.shape[0])[:, None]
    np.add.at(kernel, (indices, np.broadcast_to(source_columns, indices.shape)), weights)
    return kernel


def moment_matrices(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build total-heat and first-moment matrices in scaled coordinates."""
    source_scaled, target_scaled = scaled_coordinates(source_xyz, target_xyz)
    source_moments = np.vstack([np.ones(source_xyz.shape[0]), source_scaled.T])
    target_moments = np.vstack([np.ones(target_xyz.shape[0]), target_scaled.T])
    return source_moments, target_moments


def independent_moment_rows(target_moments: np.ndarray, *, relative_tolerance: float = 1.0e-12) -> np.ndarray:
    """Retain total heat and each numerically independent coordinate moment."""
    if target_moments.ndim != 2 or target_moments.shape[0] < 1:
        raise ValueError("target_moments must be a nonempty matrix")
    rows = [0]
    current = target_moments[rows]
    for row in range(1, target_moments.shape[0]):
        candidate = target_moments[rows + [row]]
        singular_values = np.linalg.svd(candidate, compute_uv=False)
        threshold = float(relative_tolerance) * max(float(singular_values[0]), 1.0)
        rank = int(np.sum(singular_values > threshold))
        if rank > len(rows):
            rows.append(row)
            current = candidate
    if np.linalg.matrix_rank(current) != len(rows):
        raise ValueError("the total-heat constraint is numerically degenerate")
    return np.asarray(rows, dtype=int)


def conservative_moment_operator(
    kernel: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    target_capacity: np.ndarray,
    *,
    regularization: float = 0.0,
    rank_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, float]:
    """Project a local map onto exact heat-rate and first-moment constraints.

    The returned matrix is the unique minimum correction to ``kernel`` in the
    inverse-capacity norm when the four constraint rows are independent.
    """
    source_moments, target_moments = moment_matrices(source_xyz, target_xyz)
    active_rows = independent_moment_rows(target_moments, relative_tolerance=rank_tolerance)
    source_moments = source_moments[active_rows]
    target_moments = target_moments[active_rows]
    capacity = np.asarray(target_capacity, dtype=float)
    if capacity.shape != (target_xyz.shape[0],):
        raise ValueError("target_capacity must contain one positive value per target")
    if np.any(capacity <= 0.0):
        raise ValueError("target capacities must be strictly positive")
    if kernel.shape != (target_xyz.shape[0], source_xyz.shape[0]):
        raise ValueError("kernel shape does not match source and target point counts")

    gram = (target_moments * capacity[None, :]) @ target_moments.T
    condition = float(np.linalg.cond(gram))
    scale = max(float(np.trace(gram)) / gram.shape[0], 1.0)
    gram_regularized = gram + float(regularization) * scale * np.eye(gram.shape[0])
    mismatch = source_moments - target_moments @ kernel
    correction = capacity[:, None] * target_moments.T @ np.linalg.solve(gram_regularized, mismatch)
    return kernel + correction, condition


def apply_heat_rate_operator(operator: np.ndarray, face_flux: np.ndarray, face_area: np.ndarray) -> np.ndarray:
    """Map source heat fluxes to target heat rates."""
    source_rate = np.asarray(face_flux, dtype=float) * np.asarray(face_area, dtype=float)
    if operator.shape[1] != source_rate.size:
        raise ValueError("operator and source field sizes differ")
    return operator @ source_rate


def diagnostics(
    operator: np.ndarray,
    kernel: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_rate: np.ndarray,
    condition_number: float,
) -> TransferDiagnostics:
    source_moments, target_moments = moment_matrices(source_xyz, target_xyz)
    target_rate = operator @ source_rate
    source_constraints = source_moments @ source_rate
    target_constraints = target_moments @ target_rate
    heat_scale = max(abs(float(source_constraints[0])), np.linalg.norm(source_rate), 1.0e-30)
    moment_scale = max(np.linalg.norm(source_constraints[1:]), heat_scale, 1.0e-30)
    return TransferDiagnostics(
        heat_rate_error=abs(float(target_constraints[0] - source_constraints[0])) / heat_scale,
        moment_error=float(np.linalg.norm(target_constraints[1:] - source_constraints[1:]) / moment_scale),
        correction_norm=float(np.linalg.norm(operator - kernel) / max(np.linalg.norm(kernel), 1.0e-30)),
        condition_number=condition_number,
    )
