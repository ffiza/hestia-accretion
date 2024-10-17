import unittest
import numpy as np

from hestia.tools import weighted_percentile


class WeightedPercentileTests(unittest.TestCase):
    """
    Some tests for the `weighted_percentile` function defined in `tools.py`.
    """

    def test_01(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        w = np.ones_like(x)
        q = 90
        w_mean = w.mean()
        xp = weighted_percentile(x, w, q)
        self.assertTrue(
            np.abs(w[x < xp].sum() - w.sum() * q / 100) <= 2 * w_mean)

    def test_02(self):
        x = np.random.uniform(5, size=1_000)
        w = np.ones_like(x)
        q = 90
        w_mean = w.mean()
        xp = weighted_percentile(x, w, q)
        self.assertTrue(
            np.abs(w[x < xp].sum() - w.sum() * q / 100) <= 2 * w_mean)

    def test_03(self):
        x = np.random.uniform(1.0, 50.0, size=10_000)
        w = np.random.uniform(1.0, 7000.0, size=10_000)
        q = 90
        w_mean = w.mean()
        xp = weighted_percentile(x, w, q)
        self.assertTrue(
            np.abs(w[x < xp].sum() - w.sum() * q / 100) <= 2 * w_mean)


if __name__ == '__main__':
    unittest.main()
