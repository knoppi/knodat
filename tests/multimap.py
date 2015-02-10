#!/usr/bin/env python

import sys

sys.path.insert(0, "..")

import unittest
import numpy as np

import knodat.multimap


class TestMultiMap(unittest.TestCase):
    def setUp(self):
        self.data = knodat.multimap.MultiMap("birds.dat")
        self.reduced_data = knodat.multimap.MultiMap("birds_reduced.npy")

    def test_get1(self):
        self.assertTrue(np.all(
            self.data.get("day", bird=1) ==
            np.array([np.float(x + 1) for x in range(10)])))
        self.assertTrue(np.all(
            self.data.get("day", bird=2) ==
            np.array([np.float(x + 1) for x in range(10) if x != 1])))

    def test_reduction(self):
        # Test reduction. To this end I created a reference file with the
        # correct values which should be reproduced

        # We average the birds' data over days.
        self.data.reduce(["day"], ["bird"], False, verbose=False)
        relevant_columns = ["food", "water", "distance"]
        birds = [1, 2, 3, 4, 5]
        for bird in birds:
            for column in relevant_columns:
                new = self.data.get(column, bird=bird)[0]
                ref = self.reduced_data.get(column, bird=bird)[0]

                self.assertEqual(new, ref)

    def test_length(self):
        # make sure the correct number of entries is loaded
        self.assertEqual(self.data.length(), 49)

    def test_getitem(self):
        set1 = self.data[0]
        self.data.sort("food")
        set2 = self.data[0]

        self.assertEqual(set1["bird"], 1)
        self.assertEqual(set2["bird"], 5)

    def test_setitem(self):
        reference_data = self.data.get(day=1, bird=1)[0]

        modification_set = self.data[0]
        self.assertEqual(modification_set["bird"], 1)

        modification_set["bird"] = 6
        self.data[0] = modification_set

        modified_set = self.data.get(bird=6)[0]

        self.assertEqual(reference_data[0], modified_set[0])
        self.assertEqual(reference_data[2], modified_set[2])
        self.assertEqual(reference_data[3], modified_set[3])
        self.assertEqual(reference_data[4], modified_set[4])
        self.assertNotEqual(reference_data[1], modified_set[1])

    def test_retrieve_3d_data(self):
        self.data.set_complete(False)
        self.data.set_N = 1
        x, y, z, extent = self.data.get("day", "bird", "food")

        z2 = np.array([[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                       [1.3, 0, 1.4, 1.2, 1.5, 1.1, 1.6, 1.,  1.7, 0.9],
                       [7.4, 7.5, 7.4, 7.5, 7.4, 7.5, 7.4, 7.5, 7.4, 7.5],
                       [6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.,  7.1],
                       [2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3]]
                      )

        x_ref, y_ref = np.meshgrid(range(1, 11), range(1, 6))

        self.assertTrue(np.all(x_ref == x))
        self.assertTrue(np.all(y_ref == y))
        self.assertTrue(np.all(z == z2))

        x, y, z, extent = self.data.retrieve_3d_plot_data(
            "day", "bird", "food", N=1, data_is_complete=False)
        self.assertTrue(np.all(x_ref == x))
        self.assertTrue(np.all(y_ref == y))
        self.assertTrue(np.all(z == z2))


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiMap)
    unittest.TextTestRunner(verbosity=2).run(suite)
