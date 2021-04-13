from functools import wraps
import random
from types import SimpleNamespace

import numpy

from deap import algorithms, creator, base, tools


def log(*args, **kwargs):
    print(*args, **kwargs)


# TODO: do we need to set random seed form configuration?
# Set seed for consistency of running things
numpy.random.seed(0)
random.seed(0)

# First weight is for error (minimize error), second weight is for validity
creator.create('FitnessMin', base.Fitness, weights=(-1.0))


class RCOptimizer:
    """
    Optimize a RC circuit
    """
    def __init__(self, config, C_matrix, D_matrix,
                 training_data, validation_data, test_data,
                 names, identifier=0):
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
        self.config = config
        self.parameters = SimpleNamespace(**self.config['parameters'])
        self.C_matrix = C_matrix
        self.D_matrix = D_matrix
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.bounds = self._bounds_from_config(self.config)
        self.names = names
        self.identifier = identifier
        self.best_from_grid_search = None
        self.toolbox = base.Toolbox()

        self.individual_cls = Scenario

    def register(self):
        """Register functions with the DEAP toolbox."""
        if not self.best_from_grid_search:
            log("Seeding individual generator with a best guess.")
            rc_vals_best = self.best_from_guessing().rc_vals
        else:
            log("Seeding individual generator with grid search best guess.")
            rc_vals_best = self.best_from_grid_search.rc_vals

        self.toolbox.register(
            'individual', self.individual_cls, rc_vals=rc_vals_best,
            C=self.C_matrix, D=self.D_matrix, data=self.training_data,
            rc_bounds=self.bounds)
        self.toolbox.register(
            'population', tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register('mutate', self.mutate)
        self.toolbox.register('mate', self.mate)
        self.toolbox.register('select', self.select_individuals)

    def learn(self):
        """Execute learning."""
        self.register()
        population = self.toolbox.population(n=100)
        for _ in range(self.generations):
            offspring = algorithms.varAnd(
                population, self.toolbox, cxpb=self.parameters.cxpb,
                mutpb=self.parameters.mutpb
            )
            population = self.toolbox.select(offspring, k=len(population))
            self.top10 = tools.selBest(population, k=10)

    def best_from_guessing(self):
        """
        Return a guess for the best variable values.

        The best guess is either the mean specified in configuration or 1 if the
        mean has not been defined.

        :return dict:
        """
        return {
            k: v.get('mean') or 1 for k, v in self.config['bounds'].items()
        }

    def grid_search(self):
        """
        Execute a parameter sweep over the configuration bounds and return the
        best variable combination.

        Assigns the best parameter comination to `self.best_from_grid_search`

        :return dict:
        """
        # TODO: implement
        self.best_from_grid_search = self.best_from_guessing()
        return self.best_from_guessing()


    @property
    def individual_name(self):
        # TODO: Delete this maybe, since it's unused.
        return f"individual_{self.identifier}"

    @property
    def mutate(self):
        try:
            func_name = self.parameters.mutate['function']
        except AttributeError:
            raise AttributeError(
                "Parameters config does not define `mutate` action.")
        except KeyError:
            raise KeyError("Mutate config missing function to call.")
        func = mutation_functions.get(func_name)
        if not func:
            raise ValueError(f"No mutation function named {func_name} found.")
        return lambda: func(**self.parameters.mutate.get('kwargs', {}))

    @property
    def mate(self):
        try:
            func_name = self.parameters.mate['function']
        except AttributeError:
            raise AttributeError(
                "Parameters config does not define `mate` action.")
        except KeyError:
            raise KeyError("Mate config missing function to call")
        func = crossover_functions.get(func_name)
        if not func:
            raise ValueError(f"No crossover function named {func_name} found.")
        return lambda: func(**self.parameters.mate.get('kwargs', {}))

    @property
    def select_individuals(self):
        """ Function that selects the best individual(s) of each generation. """
        try:
            func_name = self.parameters.select['function']
        except AttributeError:
            raise AttributeError(
                "Parameters config does not define `select` action.")
        except KeyError:
            raise KeyError("Mate config missing function to call")
        func = getattr(tools, func_name, None)
        if not func:
            raise ValueError(f"No selection function named {func_name} found.")
        return lambda: func(**self.parameters.mate.get('kwargs', {}))


    @classmethod
    def from_yaml(self, filepath):
        """Instantiate from a yaml"""
        # TODO: Implement
        raise NotImplementedError

    @staticmethod
    def _bounds_from_config(config):
        """ Return dictionary of tuples representing min/max bounds. """
        if 'bounds' not in config:
            return {}
        bounds = {}
        for k, v in config['bounds'].items():
            if v is None or 'pct_range' not in v:
                bounds[k] = (-numpy.inf, numpy.inf)
                continue
            mean = v['mean']
            pct_range = v['pct_range']
            rng = pct_range * mean
            lower = mean - rng / 2
            upper = mean + rng / 2
            bounds[k] = (lower, upper)
        return bounds


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
        # Maximizing fitness
        self._fitness = creator.FitnessMin()

    def predict(self):
        """Simulate the output variable based on the state matrices."""
        # For now, we just return the results as the sum of the A matrix.
        # TODO: Actually simulate the output y array
        return numpy.ones((1, 1)) * sum(self.rc_vals.values())

    def error(self):
        """Return the error between the simulated data and y array."""
        # TODO: Make this actually output the error.
        return abs(numpy.sum(self.predict()) - 20)

    @property
    def fitness(self):
        self._fitness.values = self.error(),
        return self._fitness

    @property
    def rc_valid(self):
        """Return True if all RC values are within specified bounds."""
        if not self.rc_bounds:
            return True
        for k, v in self.rc_bounds.items():
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
        if ind1.rc_bounds:
            for k, (low, high) in ind1.rc_bounds.items():
                ind1.rc_vals[k] = max(min(high, ind1.rc_vals[k]), low)
        if ind2.rc_bounds:
            for k, (low, high) in ind2.rc_bounds.items():
                ind2.rc_vals[k] = max(min(high, ind2.rc_vals[k]), low)

        return ind1, ind2
    return wrapped


def deap_mut_wrapper(func):
    """Decorator for DEAP mutation functions"""
    @wraps(func)
    def wrapped(individual, *args, **kwargs):
        keys = sorted(individual.rc_vals.keys())
        seq, = func([individual.rc_vals[k] for k in keys], *args, **kwargs)
        individual.rc_vals = {k: v for k, v in zip(keys, seq)}
        if individual.rc_bounds:
            for k, (low, high) in individual.rc_bounds.items():
                individual.rc_vals[k] = max(
                    min(high, individual.rc_vals[k]), low)
        return individual,
    return wrapped


@deap_mut_wrapper
def mutGaussianScaled(individual, mu, sigma_scale, indp):
    """
    Apply a Gaussian mutation, with the standard deviation for each attribute
    fixed as a percentage of the attribute's magnitude.

    :param float sigma_scale: scaling factor for standard deviation as a percent
        of magnitude
    """
    sigmas = [v * sigma_scale for v in individual]
    return tools.mutGaussian(individual, mu, sigmas, indp)


# Note: some of these mutations / crossovers change list size, so we will want
# to exclude them.
crossover_functions = {
    "cxBlend": deap_cx_wrapper(tools.cxBlend),
    "cxESBlend": deap_cx_wrapper(tools.cxESBlend),
    "cxESTwoPoint": deap_cx_wrapper(tools.cxESTwoPoint),
    "cxESTwoPoints": deap_cx_wrapper(tools.cxESTwoPoints),
    "cxMessyOnePoint": deap_cx_wrapper(tools.cxMessyOnePoint),
    "cxOnePoint": deap_cx_wrapper(tools.cxOnePoint),
    "cxOrdered": deap_cx_wrapper(tools.cxOrdered),
    "cxPartialyMatched": deap_cx_wrapper(tools.cxPartialyMatched),
    "cxSimulatedBinary": deap_cx_wrapper(tools.cxSimulatedBinary),
    "cxSimulatedBinaryBounded": deap_cx_wrapper(tools.cxSimulatedBinaryBounded),
    "cxTwoPoint": deap_cx_wrapper(tools.cxTwoPoint),
    "cxTwoPoints": deap_cx_wrapper(tools.cxTwoPoints),
    "cxUniform": deap_cx_wrapper(tools.cxUniform),
    "cxUniformPartialyMatched": deap_cx_wrapper(tools.cxUniformPartialyMatched)
}

mutation_functions = {
    "mutESLogNormal": deap_mut_wrapper(tools.mutESLogNormal),
    "mutFlipBit": deap_mut_wrapper(tools.mutFlipBit),
    "mutGaussian": deap_mut_wrapper(tools.mutGaussian),
    "mutPolynomialBounded": deap_mut_wrapper(tools.mutPolynomialBounded),
    "mutShuffleIndexes": deap_mut_wrapper(tools.mutShuffleIndexes),
    "mutUniformInt": deap_mut_wrapper(tools.mutUniformInt)
}

