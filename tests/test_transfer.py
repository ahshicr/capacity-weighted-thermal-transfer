from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from conservative_transfer import (
    conservative_moment_operator,
    independent_moment_rows,
    local_kernel,
    moment_matrices,
)
from run_manufactured_benchmark import (
    build_common_refinement,
    field_thermal_front,
    integrate_cells,
    make_mesh,
    project_moments,
    scaled_moment_matrices,
)


class ConservativeTransferTests(unittest.TestCase):
    def test_full_rank_operator_preserves_four_constraints(self) -> None:
        generator = np.random.default_rng(19)
        source = generator.random((80, 3))
        target = generator.random((23, 3))
        kernel = local_kernel(source, target, neighbors=6)
        capacity = 0.5 + generator.random(target.shape[0])
        operator, condition = conservative_moment_operator(
            kernel, source, target, capacity
        )
        source_moments, target_moments = moment_matrices(source, target)
        self.assertTrue(np.isfinite(condition))
        self.assertTrue(
            np.allclose(
                target_moments @ operator,
                source_moments,
                rtol=2.0e-12,
                atol=2.0e-12,
            )
        )

    def test_planar_target_drops_dependent_out_of_plane_moment(self) -> None:
        source = np.array(
            [[0.1, 0.1, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.9, 0.9, 0.0]]
        )
        target = np.array(
            [[0.2, 0.2, 0.0], [0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.8, 0.8, 0.0]]
        )
        source_moments, target_moments = moment_matrices(source, target)
        active = independent_moment_rows(target_moments)
        self.assertEqual(active.tolist(), [0, 1, 2])
        kernel = local_kernel(source, target, neighbors=3)
        operator, _ = conservative_moment_operator(
            kernel, source, target, np.ones(target.shape[0])
        )
        self.assertTrue(
            np.allclose(
                target_moments[active] @ operator,
                source_moments[active],
                rtol=2.0e-12,
                atol=2.0e-12,
            )
        )

    def test_common_refinement_covers_warped_target(self) -> None:
        source = make_mesh(32, 16, warp=0.0)
        target = make_mesh(28, 14, warp=0.035)
        _, _, _, _, coverage_error = build_common_refinement(source, target)
        self.assertLess(coverage_error, 1.0e-11)

    def test_capacity_projection_minimizes_capacity_correction(self) -> None:
        source = make_mesh(16, 8, warp=0.0)
        target = make_mesh(14, 7, warp=0.035)
        source_load, _, _ = integrate_cells(source, field_thermal_front)
        source_moments, target_moments = scaled_moment_matrices(
            source.centers, target.centers
        )
        provisional, _, _ = integrate_cells(target, field_thermal_front)
        provisional = provisional * np.linspace(0.997, 1.003, provisional.size)
        capacity = target.areas * (0.5 + 2.5 * target.centers[:, 0])
        uniform, _, _ = project_moments(
            provisional,
            source_load,
            source_moments,
            target_moments,
            np.ones(target.areas.size),
        )
        weighted, _, _ = project_moments(
            provisional,
            source_load,
            source_moments,
            target_moments,
            capacity,
        )
        weighted_energy = np.sum((weighted - provisional) ** 2 / capacity)
        uniform_energy = np.sum((uniform - provisional) ** 2 / capacity)
        self.assertLessEqual(weighted_energy, uniform_energy * (1.0 + 1.0e-12))


if __name__ == "__main__":
    unittest.main()
