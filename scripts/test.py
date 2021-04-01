import random

import optimizer

from unittest import TestCase


class TestOptimizer(TestCase):
    def setUp(self):
        random.seed(0)

    def test_cxtwopointcopy(self):
        class Ind:
            pass

        ind1 = Ind()
        ind2 = Ind()
        ind1.rc_vals = {'a': 1, 'b': 2, 'c': 3}
        ind2.rc_vals = {'a': 4, 'b': 5, 'c': 6}
        ind1, ind2 = optimizer.cxTwoPoint(ind1, ind2)
        self.assertEqual(ind1.rc_vals, {'a': 1, 'b': 2, 'c': 6})
        self.assertEqual(ind2.rc_vals, {'a': 4, 'b': 5, 'c': 3})
