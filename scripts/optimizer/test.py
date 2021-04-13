from inspect import unwrap
import random

import deap

from . import optimizer
import mock
from unittest import TestCase


class TestOptimizer(TestCase):
    def setUp(self):
        random.seed(0)

    def test_bounds_from_config(self):
        config = {'bounds': {'a': {'mean': 1, 'pct_range': 0.5}}}
        self.assertEqual(
            optimizer.RCOptimizer._bounds_from_config(config)['a'],
            (0.75, 1.25)
        )
        self.assertEqual(optimizer.RCOptimizer._bounds_from_config({}), {})

    @mock.patch.object(deap.tools, 'selectselect', mock.MagicMock())
    @mock.patch.dict(optimizer.crossover_functions,
                     {'matemate': mock.MagicMock()})
    @mock.patch.dict(optimizer.mutation_functions,
                     {'mutatemutate': mock.MagicMock()})
    def test_registration(self, _, __, select_func):
        # Test that our mutation / crossover / selection functions register and
        # get called with appropriate kwargs
        config = {
            'parameters': {
                'mate': {'function': 'matemate'}},
                'mutate': {
                    'function': 'mutatemutate', 'kwargs': {'foo': 'bar'}},
                'select': {
                    'function': {'selectselect': {'kwargs': {}}}
                }
        }


class TestDeapWrappers(TestCase):
    def setUp(self):
        random.seed(0)

    def test_cxtwopointcopy(self):
        class Ind:
            def __init__(self):
                self.rc_bounds = {}

        ind1 = Ind()
        ind2 = Ind()
        ind1.rc_vals = {'a': 1, 'b': 2, 'c': 3}
        ind2.rc_vals = {'a': 4, 'b': 5, 'c': 6}
        ind1, ind2 = optimizer.crossover_functions['cxTwoPoint'](ind1, ind2)
        self.assertEqual(ind1.rc_vals, {'a': 1, 'b': 2, 'c': 6})
        self.assertEqual(ind2.rc_vals, {'a': 4, 'b': 5, 'c': 3})

    def test_mutgaussianscaled(self):
        mutGaussianScaled = unwrap(optimizer.mutGaussianScaled)
        expect = [1.03, 0.48, 0.28, 149.22, 5.4, 45.35, 1025.3, 306.45, 1906.23]
        individual = [1, 1, 1, 100, 100, 100, 1000, 1000, 1000]
        individual = mutGaussianScaled(individual, 0, 0.5, indp=1)[0]
        individual = [round(i, 2) for i in individual]
        self.assertListEqual(
            individual, expect
        )
