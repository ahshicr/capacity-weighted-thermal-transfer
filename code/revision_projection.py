"""Projection identities and a checked optional nonnegative extension."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import LinearConstraint, linprog, minimize


def validate_problem(prior, capacity, features, moments):
    prior = np.asarray(prior, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    features = np.asarray(features, dtype=float)
    moments = np.asarray(moments, dtype=float)
    if prior.ndim != 1 or capacity.shape != prior.shape:
        raise ValueError("prior and capacity must be vectors of equal length")
    if features.ndim != 2 or features.shape[1] != prior.size:
        raise ValueError("features must have one column per target")
    if moments.shape != (features.shape[0],):
        raise ValueError("moments must have one entry per retained feature")
    if any(not np.all(np.isfinite(array)) for array in (prior, capacity, features, moments)):
        raise ValueError("all inputs must be finite")
    if np.any(capacity <= 0):
        raise ValueError("capacities must be strictly positive")
    if np.linalg.matrix_rank(features) != features.shape[0]:
        raise ValueError("retained feature rows must be independent")
    return prior, capacity, features, moments


def capacity_projection(prior, capacity, features, moments):
    prior, capacity, features, moments = validate_problem(prior, capacity, features, moments)
    gram = (features * capacity) @ features.T
    factor = cho_factor(gram, lower=True, check_finite=False)
    multiplier = cho_solve(factor, moments - features @ prior, check_finite=False)
    return prior + capacity * (features.T @ multiplier)


def error_decomposition(prior, exact, capacity, features, moments):
    """Return the two orthogonal terms for possibly incompatible exact loads."""
    prior, capacity, features, moments = validate_problem(prior, capacity, features, moments)
    exact = np.asarray(exact, dtype=float)
    if exact.shape != prior.shape or not np.all(np.isfinite(exact)):
        raise ValueError("exact must be a finite target-load vector")
    factor = cho_factor((features * capacity) @ features.T, lower=True, check_finite=False)
    initial_error = prior - exact
    kernel_error = initial_error - capacity * (
        features.T @ cho_solve(factor, features @ initial_error, check_finite=False)
    )
    defect = moments - features @ exact
    compatibility_error = capacity * (
        features.T @ cho_solve(factor, defect, check_finite=False)
    )
    return kernel_error, compatibility_error, defect


def nonnegative_capacity_projection(prior, capacity, features, moments):
    """Solve a feasible nonnegative QP and check its primal and dual certificate.

    Already nonnegative equality projections use the original closed form.
    Other cases require a feasibility LP followed by SLSQP. This optional
    extension is not claimed to have the cost of one small Gram solve.
    """
    prior, capacity, features, moments = validate_problem(prior, capacity, features, moments)
    closed = capacity_projection(prior, capacity, features, moments)
    if np.min(closed) >= 0.0:
        return closed, {"path": "closed_form", "iterations": 0}

    feasible = linprog(
        np.zeros(prior.size), A_eq=features, b_eq=moments,
        bounds=(0.0, None), method="highs",
    )
    if not feasible.success:
        if feasible.status == 2:
            raise ValueError("nonnegative loads cannot satisfy the requested moments")
        raise RuntimeError(f"feasibility check failed: {feasible.message}")

    def objective(load):
        difference = load - prior
        return 0.5 * float(np.sum(difference * difference / capacity))

    def gradient(load):
        return (load - prior) / capacity

    result = minimize(
        objective, feasible.x, jac=gradient, method="SLSQP",
        bounds=[(0.0, None)] * prior.size,
        constraints=LinearConstraint(features, moments, moments),
        options={"ftol": 1.0e-13, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"nonnegative optimization failed: {result.message}")
    load = result.x
    scale = max(1.0, float(np.max(np.abs(moments))))
    if np.min(load) < 0.0 or np.max(np.abs(features @ load - moments)) > 1.0e-10 * scale:
        raise RuntimeError("nonnegative result failed the primal feasibility check")

    # Certify a KKT multiplier even when the free columns do not have full rank.
    active = load <= 1.0e-10 * max(1.0, float(np.max(load)))
    free = ~active
    grad = gradient(load)
    dual = linprog(
        np.zeros(features.shape[0]),
        A_ub=features[:, active].T if np.any(active) else None,
        b_ub=grad[active] if np.any(active) else None,
        A_eq=features[:, free].T if np.any(free) else None,
        b_eq=grad[free] if np.any(free) else None,
        bounds=[(None, None)] * features.shape[0], method="highs",
    )
    if not dual.success:
        raise RuntimeError("nonnegative result failed the dual optimality check")
    reduced_gradient = grad - features.T @ dual.x
    tolerance = 1.0e-8 * max(1.0, float(np.max(np.abs(grad))))
    if np.any(np.abs(reduced_gradient[free]) > tolerance) or np.any(reduced_gradient[active] < -tolerance):
        raise RuntimeError("nonnegative result failed the KKT optimality check")
    return load, {
        "path": "constrained_qp", "iterations": int(result.nit),
        "active_bounds": int(np.count_nonzero(active)),
        "maximum_equality_residual": float(np.max(np.abs(features @ load - moments))),
        "maximum_free_stationarity_residual": float(np.max(np.abs(reduced_gradient[free]))) if np.any(free) else 0.0,
        "minimum_active_reduced_gradient": float(np.min(reduced_gradient[active])) if np.any(active) else 0.0,
    }
