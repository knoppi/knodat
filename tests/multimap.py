#!/usr/bin/env python

import sys

sys.path.insert(0, "..")

import unittest
import numpy as np

import multimap


class TestMultiMap(unittest.TestCase):
    def setUp(self):
        self.data = multimap.MultiMap("birds.dat")
        self.reduced_data = multimap.MultiMap("birds_reduced.npy")

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

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiMap)
    unittest.TextTestRunner(verbosity=2).run(suite)
