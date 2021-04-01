from functools import wraps
import random

import numpy

from deap import creator, base, tools, algorithms

# TODO: change random seed
numpy.random.seed(0)
random.seed(0)

toolbox = base.Toolbox()


class RCOptimizer:
    """
    Optimize a RC circuit
    """
    def __init__(self, B_matrix, C_matrix, D_matrix,
                 training_data, validation_data, test_data, bounds,
                 names, n_individuals=100, generations=50, identifier=0):
        """Optimize a matrix of RC values.

        Assumes the governing equations are of the form `xdot = A x + B u`
        and `y = C xdot + D u`

        :param ndarray B_matrix:
        :param ndarray C_matrix:
        :param ndarray D_matrix:
        :param SimulationData training_data:
        :param SimulationData validation_data:
        :param SimulationData test_data:
        :param dict bounds: bounds for entries in the A matrix; should be
            with keys being the names of the R / C variables and
            values being a tuple of the lower & upper bounds
        :param dict names: Mapping of numerical R & C values to English names
        :param float timestep: Simulation timestep
        :param int identifier: ID for this optimization
        """
        self.B_matrix = B_matrix
        self.C_matrix = C_matrix
        self.D_matrix = D_matrix
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.bounds = bounds
        self.names = names
        self.n_individuals = n_individuals
        self.generations = generations
        self.identifier = identifier

        self.validate_bounds()

    def register(self):
        """Register functions with the DEAP toolbox."""
        toolbox.register(
            f"individual_{self.identifier}", tools.initRepeat,
            creator.Individual, A=self.A, B=self.B_matrix, C=self.C_matrix,
            D=self.D_matrix, data=self.training_data, n=self.n_individuals)
        toolbox.register('population', tools.initRepeat, list, toolbox)

    def learn(self):
        """Execute learning."""
        pass

    @property
    def A(self):
        return numpy.random.rand((3, 3))

    @property
    def individual_name(self):
        return f"individual_{self.identifier}"

    def validate_bounds(self):
        for v in self.bounds.values():
            if v[0] > v[1]:
                raise ValueError(
                    "First element of bound should be smaller than the second"
                    "element."
                )


class Scenario:
    """Simulate a building state with given state matrices."""

    def __init__(self, rc_vals, C, D, data, rc_bounds=None):
        """
        :param dict rc_vals: RC values
        :param SimulationData data:
        :param dict rc_bounds: Dictionary of tuples bounding each RC value.
        """
        self.rc_vals = rc_vals
        self.C = C
        self.D = D
        self.data = data
        self.rc_bounds = rc_bounds

    def predict(self):
        """Simulate the output variable based on the state matrices."""
        # For now, we just return the results as the sum of the A matrix.
        # TODO: Actually simulate the output y array
        return numpy.ones((self.steps, 1)) * sum(self.rc_vals.values())

    def error(self):
        """Return the error between the simulated data and y array."""
        # TODO: Make this actually output the error.
        return sum(self.predict())

    def fitness(self):
        """Return the fitness of the individual"""
        if not self.rc_valid:
            return 0
        # TODO: Make this more efficient..? Division is slow.
        return 1 / self.error()

    @property
    def rc_valid(self):
        """Return True if all RC values are within specified bounds."""
        if not self.rc_bounds:
            return True
        for k, v in self.rc_bounds.values():
            low = v[0]
            high = v[1]
            val = self.rc_vals[k]
            if val < low or val > high:
                return False
        return True


class SimulationData:
    """Simulation data container."""

    def __init__(self, x0, u, timestep, y=None):
        self.x0 = x0
        self.u = u
        self.timestep = timestep
        self.y = y
        self.steps = u.shape[0]
        if y is not None and y.shape[0] != self.steps:
            raise ValueError(
                "Mismatch in u and y array rows sizes: "
                f"{self.steps} != {y.shape[0]}"
            )


def deap_cx_wrapper(func):
    """Decorator for DEAP crossover functions"""
    @wraps(func)
    def wrapped(ind1, ind2, *args, **kwargs):
        keys = sorted(ind1.rc_vals.keys())
        seq1, seq2 = func(
            [ind1.rc_vals[k] for k in keys],
            [ind2.rc_vals[k] for k in keys],
            *args, **kwargs
        )
        ind1.rc_vals = {k: v for k, v in zip(keys, seq1)}
        ind2.rc_vals = {k: v for k, v in zip(keys, seq2)}
        return ind1, ind2
    return wrapped


def deap_mut_wrapper(func):
    """Decorator for DEAP mutation functions"""
    @wraps(func)
    def wrapped(individual, *args, **kwargs):
        keys = sorted(individual.rc_vals.keys())
        seq = func([individual.rc_vals[k] for k in keys], *args, **kwargs)
        individual.rc_vals = {k: v for k, v in zip(keys, seq)}
        return individual
    return wrapped


cxBlend = deap_cx_wrapper(tools.cxBlend)
cxESBlend = deap_cx_wrapper(tools.cxESBlend)
cxESTwoPoint = deap_cx_wrapper(tools.cxESTwoPoint)
cxESTwoPoints = deap_cx_wrapper(tools.cxESTwoPoints)
cxMessyOnePoint = deap_cx_wrapper(tools.cxMessyOnePoint)
cxOnePoint = deap_cx_wrapper(tools.cxOnePoint)
cxOrdered = deap_cx_wrapper(tools.cxOrdered)
cxPartialyMatched = deap_cx_wrapper(tools.cxPartialyMatched)
cxSimulatedBinary = deap_cx_wrapper(tools.cxSimulatedBinary)
cxSimulatedBinaryBounded = deap_cx_wrapper(tools.cxSimulatedBinaryBounded)
cxTwoPoint = deap_cx_wrapper(tools.cxTwoPoint)
cxTwoPoints = deap_cx_wrapper(tools.cxTwoPoints)
cxUniform = deap_cx_wrapper(tools.cxUniform)
cxUniformPartialyMatched = deap_cx_wrapper(tools.cxUniformPartialyMatched)

mutESLogNormal = deap_mut_wrapper(tools.mutESLogNormal)
mutFlipBit = deap_mut_wrapper(tools.mutFlipBit)
mutGaussian = deap_mut_wrapper(tools.mutGaussian)
mutPolynomialBounded = deap_mut_wrapper(tools.mutPolynomialBounded)
mutShuffleIndexes = deap_mut_wrapper(tools.mutShuffleIndexes)
mutUniformInt = deap_mut_wrapper(tools.mutUniformInt)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Scenario, fitness=creator.FitnessMax)
toolbox.register("mate", cxTwoPoint)
