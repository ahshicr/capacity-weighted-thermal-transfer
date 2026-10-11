from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from analyze_transfer import target_inward_heat_flux, _thermal_response
from conservative_transfer import apply_heat_rate_operator


class ThermalFluxSignTests(unittest.TestCase):
    def test_hot_fluid_heats_target_after_interface_conversion(self):
        # Fluid is 20 K hotter across 0.1 m, with conductivity 0.5 W/(m K).
        # The fluid outward normal points toward the colder solid wall.
        exported = np.array([0.5 * (300.0 - 320.0) / 0.1])
        inward = target_inward_heat_flux(exported)
        load = apply_heat_rate_operator(np.ones((1, 1)), inward, np.array([0.2]))
        np.testing.assert_allclose(load, [20.0], atol=1e-12)
        temperature = _thermal_response(load, np.array([100.0]), np.zeros(1),
                                        sparse.csr_matrix((1, 1)), 10.0, 1.0)
        np.testing.assert_allclose(temperature, [2.0], atol=1e-12)

    def test_cooling_has_the_opposite_target_response(self):
        load = target_inward_heat_flux(np.array([100.0])) * 0.2
        temperature = _thermal_response(load, np.array([100.0]), np.zeros(1),
                                        sparse.csr_matrix((1, 1)), 10.0, 1.0)
        np.testing.assert_allclose(temperature, [-2.0], atol=1e-12)

    def test_already_inward_data_are_not_flipped_twice(self):
        values = np.array([-2.0, 3.0])
        np.testing.assert_array_equal(target_inward_heat_flux(values, 'target-inward'), values)
        with self.assertRaises(ValueError):
            target_inward_heat_flux(values, 'unspecified')


if __name__ == '__main__':
    unittest.main()
