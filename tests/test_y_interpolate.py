import unittest
import numpy as np

from hestia.tools import y_interpolate


class Tests(unittest.TestCase):
    def test_01(self):
        x = y_interpolate(
            y=0.5,
            xp=np.array([0, 1]),
            yp=np.array([0, 1])
        )
        self.assertEqual(len(x), 1)
        self.assertAlmostEqual(x[0], 0.5)

    def test_02(self):
        x = y_interpolate(
            y=0.5,
            xp=np.array([0, 2]),
            yp=np.array([1, -1])
        )
        self.assertEqual(len(x), 1)
        self.assertAlmostEqual(x[0], 0.5)

    def test_03(self):
        x = y_interpolate(
            y=1,
            xp=np.array([0, 1, 2, 3, 4]),
            yp=np.array([0.5, 1.5, 0.75, 2, 2.5])
        )
        self.assertEqual(len(x), 3)
        self.assertAlmostEqual(x[0], 0.5)
        self.assertAlmostEqual(x[1], 5/3)
        self.assertAlmostEqual(x[2], 2.2)


if __name__ == '__main__':
    unittest.main()
