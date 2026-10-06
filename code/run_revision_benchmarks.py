"""Additional, separately recorded numerical evidence for the minor revision."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.linalg import cho_factor, cho_solve

from revision_projection import capacity_projection, error_decomposition, nonnegative_capacity_projection
from run_manufactured_benchmark import (
    FIELDS, apply_common_refinement, build_common_refinement,
    integrate_cells, make_mesh, make_thermal_model, scaled_moment_matrices,
)


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_rbf(source, values):
    return RBFInterpolator(
        source.centers, values, neighbors=min(32, len(source.centers)),
        smoothing=1.0e-10, kernel="thin_plate_spline", degree=1,
    )


def weighted_squared(vector, capacity):
    return float(np.sum(vector * vector / capacity))


def timed(function, repeats, batch=1, warmup=2):
    for _ in range(warmup):
        function()
    measurements = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(batch):
                function()
            measurements.append((time.perf_counter_ns() - started) / (1.0e9 * batch))
    finally:
        if enabled:
            gc.enable()
    return measurements


def timing_benchmark(setup_repeats, apply_repeats, small_batch):
    summary, raw = [], []
    grids = [((16, 8), (14, 7)), ((32, 16), (28, 14)), ((64, 32), (56, 28)), ((128, 64), (56, 28))]
    for number, (source_shape, target_shape) in enumerate(grids, start=1):
        source, target = make_mesh(*source_shape, warp=0.0), make_mesh(*target_shape, warp=0.035)
        source_load, _, _ = integrate_cells(source, FIELDS["thermal_front"])
        source_flux = source_load / source.areas
        source_moments, target_moments = scaled_moment_matrices(source.centers, target.centers)
        capacity = make_thermal_model(target).capacity
        rows, columns, overlap, _, coverage = build_common_refinement(source, target)
        if coverage > 1e-11:
            raise RuntimeError("common-refinement coverage check failed")
        factor = cho_factor((target_moments * capacity) @ target_moments.T, lower=True, check_finite=False)
        provisional = make_rbf(source, source_flux)(target.centers) * target.areas
        residual = source_moments @ source_load - target_moments @ provisional
        multiplier = cho_solve(factor, residual, check_finite=False)

        def prior_update():
            # Construct with the current flux values; no preassembled RBF map.
            return make_rbf(source, source_flux)(target.centers) * target.areas

        def full_projected_update():
            initial = prior_update()
            difference = source_moments @ source_load - target_moments @ initial
            dual = cho_solve(factor, difference, check_finite=False)
            return initial + capacity * (target_moments.T @ dual)

        stages = {
            "gram_setup": (lambda: cho_factor((target_moments * capacity) @ target_moments.T, lower=True, check_finite=False), setup_repeats, small_batch, "setup"),
            "common_setup": (lambda: build_common_refinement(source, target), setup_repeats, 1, "setup"),
            "rbf_prior": (prior_update, apply_repeats, 1, "application"),
            "residual": (lambda: source_moments @ source_load - target_moments @ provisional, apply_repeats, small_batch, "application"),
            "gram_solve": (lambda: cho_solve(factor, residual, check_finite=False), apply_repeats, small_batch, "application"),
            "correction_update": (lambda: provisional + capacity * (target_moments.T @ multiplier), apply_repeats, small_batch, "application"),
            "projected_total": (full_projected_update, apply_repeats, 1, "application"),
            "common_apply": (lambda: apply_common_refinement(rows, columns, overlap, source_flux, len(target.areas)), apply_repeats, small_batch, "application"),
        }
        samples_by_stage = {}
        for name, (function, repeats, batch, phase) in stages.items():
            if phase == "setup":
                samples_by_stage[name] = timed(function, repeats, batch=batch)
        application_names = [name for name, value in stages.items() if value[3] == "application"]
        for name in application_names:
            samples_by_stage[name] = []
            for _ in range(2):
                stages[name][0]()
        # Alternate stage order to avoid comparing a cold first stage with
        # a later stage under a different sustained CPU load or clock rate.
        for repeat in range(apply_repeats):
            offset = repeat % len(application_names)
            order = application_names[offset:] + application_names[:offset]
            for name in order:
                function, _, batch, _ = stages[name]
                samples_by_stage[name].extend(timed(function, 1, batch=batch, warmup=0))
        record = {"case": f"T{number}", "source_cells": len(source.areas), "target_cells": len(target.areas), "constraints": target_moments.shape[0]}
        for name, (function, repeats, batch, phase) in stages.items():
            samples = samples_by_stage[name]
            for repeat, elapsed in enumerate(samples, start=1):
                raw.append({"case": record["case"], "stage": name, "phase": phase, "repeat": repeat, "batch": batch, "mean_elapsed_s": elapsed})
            record[name + "_median_ms"] = 1e3 * float(np.median(samples))
            record[name + "_q25_ms"] = 1e3 * float(np.quantile(samples, 0.25))
            record[name + "_q75_ms"] = 1e3 * float(np.quantile(samples, 0.75))
        projected = full_projected_update()
        desired = source_moments @ source_load
        record["relative_constraint_residual"] = float(np.linalg.norm(target_moments @ projected - desired) / abs(desired[0]))
        record["projection_component_percent_of_prior"] = 100 * sum(
            record[name + "_median_ms"] for name in ("residual", "gram_solve", "correction_update")
        ) / record["rbf_prior_median_ms"]
        summary.append(record)
        print(f"timing {record['case']}: {len(source.areas)} to {len(target.areas)} cells", flush=True)
    return summary, raw


def moment_compatibility_audit():
    records = []
    for source_shape in ((16, 8), (32, 16), (64, 32), (128, 64)):
        for target_shape in ((14, 7), (28, 14), (56, 28)):
            source, target = make_mesh(*source_shape, warp=0.0), make_mesh(*target_shape, warp=0.035)
            source_moments, target_moments = scaled_moment_matrices(source.centers, target.centers)
            capacity = make_thermal_model(target).capacity
            for name, field in FIELDS.items():
                source_load, _, _ = integrate_cells(source, field)
                exact, _, _ = integrate_cells(target, field)
                prior = make_rbf(source, source_load / source.areas)(target.centers) * target.areas
                desired = source_moments @ source_load
                projected = capacity_projection(prior, capacity, target_moments, desired)
                kernel, compatibility, defect = error_decomposition(prior, exact, capacity, target_moments, desired)
                truth_scale = max(weighted_squared(exact, capacity), 1e-30)
                final_error = weighted_squared(projected - exact, capacity)
                decomposed = weighted_squared(kernel, capacity) + weighted_squared(compatibility, capacity)
                feasible_reference = exact + compatibility
                pythagorean_initial = weighted_squared(feasible_reference - prior, capacity)
                pythagorean_final = weighted_squared(feasible_reference - projected, capacity)
                correction = weighted_squared(projected - prior, capacity)
                record = {
                    "source_nx": source_shape[0], "source_ny": source_shape[1],
                    "target_nx": target_shape[0], "target_ny": target_shape[1], "field": name,
                    "relative_total_defect": float(abs(defect[0]) / abs(desired[0])),
                    "first_moment_defect_relative_to_total": float(np.linalg.norm(defect[1:]) / abs(desired[0])),
                    "prior_error_squared_capacity_metric": weighted_squared(prior - exact, capacity),
                    "projected_error_squared_capacity_metric": final_error,
                    "kernel_error_squared_capacity_metric": weighted_squared(kernel, capacity),
                    "compatibility_error_squared_capacity_metric": weighted_squared(compatibility, capacity),
                    "decomposition_residual_relative_to_truth_energy": abs(final_error - decomposed) / truth_scale,
                    "pythagorean_residual_relative_to_truth_energy": abs(pythagorean_initial - pythagorean_final - correction) / truth_scale,
                    "feasible_reference_constraint_residual_relative_to_total": float(np.linalg.norm(target_moments @ feasible_reference - desired) / abs(desired[0])),
                }
                if record["decomposition_residual_relative_to_truth_energy"] > 1e-12:
                    raise RuntimeError("moment-defect decomposition check failed")
                if record["pythagorean_residual_relative_to_truth_energy"] > 1e-12:
                    raise RuntimeError("conditional Pythagorean check failed")
                records.append(record)
        print(f"moment audit: source {source_shape[0]} by {source_shape[1]}", flush=True)
    assert len(records) == 48
    return records


def nonnegative_examples(repeats):
    features = np.array([[1., 1., 1.], [0., 0.5, 1.]])
    prior = np.full(3, 1.0 / 3.0)
    capacity = np.ones(3)
    records = []
    for case, location in (("interior", 0.5), ("near_boundary", 0.95), ("outside_hull", 1.1)):
        desired = np.array([1.0, location])
        closed = capacity_projection(prior, capacity, features, desired)
        closed_times = timed(lambda: capacity_projection(prior, capacity, features, desired), repeats, batch=128)
        record = {
            "case": case, "requested_total_W": 1.0, "requested_first_moment_W_m": location,
            "closed_load_0_W": float(closed[0]), "closed_load_1_W": float(closed[1]), "closed_load_2_W": float(closed[2]),
            "closed_minimum_W": float(np.min(closed)),
            "closed_maximum_constraint_residual": float(np.max(np.abs(features @ closed - desired))),
            "closed_median_runtime_ms": 1e3 * float(np.median(closed_times)),
        }
        try:
            bounded, diagnostics = nonnegative_capacity_projection(prior, capacity, features, desired)
        except ValueError as error:
            if case != "outside_hull":
                raise
            record.update({"nonnegative_feasible": False, "bounded_load_0_W": "", "bounded_load_1_W": "", "bounded_load_2_W": "", "bounded_maximum_constraint_residual": "", "bounded_median_runtime_ms": "", "solver_path": "infeasible", "iterations": ""})
        else:
            if case == "outside_hull":
                raise RuntimeError("outside-hull example was incorrectly accepted")
            bounded_times = timed(lambda: nonnegative_capacity_projection(prior, capacity, features, desired), repeats)
            record.update({
                "nonnegative_feasible": True,
                "bounded_load_0_W": float(bounded[0]), "bounded_load_1_W": float(bounded[1]), "bounded_load_2_W": float(bounded[2]),
                "bounded_maximum_constraint_residual": float(np.max(np.abs(features @ bounded - desired))),
                "bounded_median_runtime_ms": 1e3 * float(np.median(bounded_times)),
                "solver_path": diagnostics["path"], "iterations": diagnostics["iterations"],
            })
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "revision_20261006")
    parser.add_argument("--setup-repeats", type=int, default=7)
    parser.add_argument("--apply-repeats", type=int, default=31)
    parser.add_argument("--small-batch", type=int, default=128)
    parser.add_argument("--cpu-label", default="unspecified")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new result directory for each revision run.")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    config = {
        "created_at_utc": started, "status": "running", "setup_repeats": args.setup_repeats,
        "apply_repeats": args.apply_repeats, "small_operation_batch": args.small_batch,
        "warmup_operations": 2, "timing_clock": "perf_counter_ns", "randomization": "none; deterministic manufactured fields",
        "application_stage_order": "cyclically interleaved across repetitions to reduce temporal drift",
        "cpu_label": args.cpu_label, "logical_processors": os.cpu_count(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "thread_limits": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "timing_scope": {
            "rbf_prior": "Current-field RBF construction, local evaluation and target-area multiplication; no cached dense prior operator.",
            "gram_setup": "Gram matrix construction and Cholesky factorization, once per fixed target geometry/capacity set.",
            "common_setup": "Intersection search and polygon clipping, once per fixed pair of cell meshes.",
            "common_apply": "Overlap accumulation with cached geometry for each source field.",
            "projection_components": "Source/target residual accumulation, cached-factor solve, and correction accumulation are timed separately.",
            "projected_total": "Full current-field RBF construction/evaluation plus projection, with the small Gram factor cached.",
        },
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), ROOT / "code" / "revision_projection.py", ROOT / "code" / "run_manufactured_benchmark.py")},
    }
    try:
        from threadpoolctl import threadpool_info
        config["blas_threads"] = [{"internal_api": item["internal_api"], "num_threads": item["num_threads"]} for item in threadpool_info()]
    except ImportError:
        config["blas_threads"] = "threadpoolctl unavailable; thread environment recorded"
    (output / "run_manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        timing, raw = timing_benchmark(args.setup_repeats, args.apply_repeats, args.small_batch)
        compatibility = moment_compatibility_audit()
        nonnegative = nonnegative_examples(args.apply_repeats)
        for filename, records in (("timing_summary.csv", timing), ("timing_raw.csv", raw), ("moment_compatibility.csv", compatibility), ("nonnegative_examples.csv", nonnegative)):
            write_csv(output / filename, records)
        summary = {
            "timing_cases": len(timing), "manufactured_cases_checked": len(compatibility),
            "maximum_relative_total_defect": max(item["relative_total_defect"] for item in compatibility),
            "minimum_first_moment_defect_relative_to_total": min(item["first_moment_defect_relative_to_total"] for item in compatibility),
            "maximum_first_moment_defect_relative_to_total": max(item["first_moment_defect_relative_to_total"] for item in compatibility),
            "maximum_decomposition_residual_relative_to_truth_energy": max(item["decomposition_residual_relative_to_truth_energy"] for item in compatibility),
            "maximum_pythagorean_residual_relative_to_truth_energy": max(item["pythagorean_residual_relative_to_truth_energy"] for item in compatibility),
            "nonnegative_cases": nonnegative,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(output.iterdir()) if path.name != "run_manifest.json"}})
    except Exception as error:
        config.update({"status": "failed", "error": str(error)})
        raise
    finally:
        (output / "run_manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
