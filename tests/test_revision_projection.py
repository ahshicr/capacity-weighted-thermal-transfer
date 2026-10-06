from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from revision_projection import capacity_projection, error_decomposition, nonnegative_capacity_projection


class RevisionProjectionTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
        self.capacity = np.array([1.0, 2.0, 4.0])
        self.prior = np.array([0.4, 0.3, 0.3])

    def test_feasible_exact_load_has_pythagorean_error_reduction(self):
        exact = np.array([0.1, 0.2, 0.7])
        moments = self.features @ exact
        projected = capacity_projection(self.prior, self.capacity, self.features, moments)
        initial = np.sum((self.prior - exact) ** 2 / self.capacity)
        final = np.sum((projected - exact) ** 2 / self.capacity)
        correction = np.sum((projected - self.prior) ** 2 / self.capacity)
        self.assertAlmostEqual(initial, final + correction, places=13)
        self.assertLessEqual(final, initial)

    def test_incompatible_exact_load_has_orthogonal_defect_decomposition(self):
        exact = np.array([0.1, 0.2, 0.7])
        moments = np.array([1.0, 0.6])
        projected = capacity_projection(self.prior, self.capacity, self.features, moments)
        kernel, compatibility, defect = error_decomposition(
            self.prior, exact, self.capacity, self.features, moments
        )
        np.testing.assert_allclose(projected - exact, kernel + compatibility, atol=1e-14)
        self.assertAlmostEqual(float(np.sum(kernel * compatibility / self.capacity)), 0.0, places=13)
        self.assertGreater(np.linalg.norm(defect), 0)
        self.assertAlmostEqual(
            float(np.sum((projected - exact) ** 2 / self.capacity)),
            float(np.sum(kernel ** 2 / self.capacity) + np.sum(compatibility ** 2 / self.capacity)), places=13,
        )

    def test_nonnegative_extension_agrees_with_known_constrained_optimum(self):
        capacity = np.ones(3)
        prior = np.full(3, 1.0 / 3.0)
        moments = np.array([1.0, 0.95])
        closed = capacity_projection(prior, capacity, self.features, moments)
        self.assertLess(np.min(closed), 0)
        bounded, diagnostics = nonnegative_capacity_projection(prior, capacity, self.features, moments)
        np.testing.assert_allclose(bounded, [0.0, 0.1, 0.9], atol=1e-11)
        np.testing.assert_allclose(self.features @ bounded, moments, atol=1e-12)
        self.assertEqual(diagnostics["path"], "constrained_qp")

    def test_nonnegative_extension_rejects_infeasible_centroid(self):
        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            nonnegative_capacity_projection(self.prior, self.capacity, self.features, [1.0, 1.1])

    def test_nonnegative_closed_form_needs_no_iterative_optimization(self):
        moments = self.features @ self.prior
        load, diagnostics = nonnegative_capacity_projection(self.prior, self.capacity, self.features, moments)
        np.testing.assert_allclose(load, self.prior, atol=1e-14)
        self.assertEqual(diagnostics["path"], "closed_form")

    def test_projection_accepts_consistent_region_features(self):
        x = np.linspace(0.0, 1.0, 6)
        features = np.vstack([np.ones(6), x, (x <= 0.4).astype(float)])
        exact = np.array([0.1, 0.2, 0.1, 0.25, 0.2, 0.15])
        prior = np.full(6, 1.0 / 6)
        projected = capacity_projection(prior, np.linspace(1.0, 2.0, 6), features, features @ exact)
        np.testing.assert_allclose(features @ projected, features @ exact, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
