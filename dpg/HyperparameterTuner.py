'''
Code taken from scikit grid search
Authors:
  Alexandre Gramfort <alexandre.gramfort@inria.fr>,
  Gael Varoquaux <gael.varoquaux@normalesup.org>
  Andreas Mueller <amueller@ais.uni-bonn.de>
  Olivier Grisel <olivier.grisel@ensta.org>

License: 
  BSD 3 clause

Adapted for my purpose: 
  Hyper Paramter Tuning with RL agents
  No training data set with labels is available, hence there is nothing to feed the "fit" function 
  The scoring of the choosen parameters depends on the world and is calculated on the fly, there is no score function for cross validation
'''

# Mapping is the base class for many containers such as dict.
from collections import Mapping
# Cartesian product of intput sequences
# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
from itertools import product
# reduce : same as built-in reduce, just included in functools for
#          compatibility with python 3
# partial: similar functionality to lambda, but allows different order of
#          arguments.  ALso slighty faster
from functools import partial, reduce
# The operator module exports a set of efficient functions corresponding to the
# intrinsic operators of Python
import operator
# Joblib
# https://pypi.python.org/pypi/joblib
# Joblib is a set of tools to provide lightweight pipelining in Python.  In
# particular, joblib offers:
#   - transparent disk-caching of the output values and lazy re-evaluation
#     (memoize pattern)
#   - easy simple parallel computing
#   - logging and tracing of the execution
# Install with pip: pip install joblib
# Install from git repository: python setup.py install
# Example for Parallel execution
# >>> from joblib import Parallel, delayed
# >>> from math import sqrt
# >>> Parallel(n_jobs=1)(delayed(sqrt) (i**2) for i in range(10) )
# >>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
from joblib import Parallel, delayed
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ParameterGrid(object):
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
      The parameter grid to explore, as a dictionary mapping estimator
      parameters to sequences of allowed values.

      An empty dict signifies default parameters.

      A sequence of dicts signifies a sequence of grids to search, and is
      useful to avoid exploring parameter combinations that make no sense
      or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.grid_search import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...  [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...   {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                {'kernel': 'rbf', 'gamma': 1},
    ...                {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See also
    --------
    :class:`GridSearchCV`:
      uses ``ParameterGrid`` to perform a full parallelized parameter search.
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        # end if
        self.param_grid = param_grid
        return

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of string to any
          Yields dictionaries mapping each estimator parameter to one of its
          allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
                    # end for
                    # end if
        # end for
        return

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1 for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
          The iteration index

        Returns
        -------
        params : dict of string to any
          Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue
                    # end
            # end

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                # end for
                return out
                # end if
        # end for
        raise IndexError('ParameterGrid index out of range')


class ParameterSampler(object):
    """
    Generator on parameters sampled from given distributions.

    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    I modified this class:
      Scipy distributions are no longer supported, instead I use numpy distributions
      which are packed in a lambda functions to generated random numbers of the given
      distribution, e.g.: x = lambda : np.random.uniform(0,1)

    Parameters
    ----------
    param_distributions : dict string to sequence or numpy distributions
      Dictionary where the keys are parameters and values
      are distributions from which a parameter is to be sampled.

      Distributions can either be:
      - a function without arguments, that returns a single number upon calling, e.g:
        x = lambda : np.random.uniform(0,1)
        x()
      - a sequence of values to sample fromm where a uniform distribution is assumed

    n_iter : integer
      Number of parameter settings that are produced.

    Iterator
    -------
    params : dict of string to any
      **Yields** dictionaries mapping each estimator parameter to
      as sampled value.

    Examples
    --------
    >>> import ParameterSampler as ParameterSampler
    >>> import numpy as np
    >>> search_space = {'a':[1, 2], 'b': lambda: np.random.uniform(0,1)}
    >>>
    >>> x = ParameterSampler(search_space,3)
    >>> for y in x: print y

        {'a': 1, 'b': 0.6414}
        {'a': 2, 'b': 0.2727}
        {'a': 1, 'b': 0.0714}
    """

    def __init__(self, param_distributions, n_iter):
        self.param_distributions = param_distributions
        self.n_iter = n_iter

        # check if all distributions are given as lists
        # in this case we want to sample without replacement
        self.all_lists = np.all([self.is_list(v) for v in self.param_distributions.values()])

        return

    def __iter__(self):
        '''
        Provides iterator functionality to allow a loop over parameter combinations
        '''

        if self.all_lists:
            # look up sampled parameter settings in parameter grid
            param_grid = ParameterGrid(self.param_distributions)
            grid_size = len(param_grid)

            if grid_size < self.n_iter:
                raise ValueError("The total space of parameters %d is smaller than n_iter=%d." % (
                grid_size, self.n_iter) + " For exhaustive searches, use GridSearchCV.")
            # end if

            # for i in sample_without_replacement(grid_size, self.n_iter, random_state=rnd):
            for i in np.random.choice(grid_size, self.n_iter, replace=False):
                # Stores current state of function call, hence choice without replacement works
                yield param_grid[i]
                # end for

        else:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(self.param_distributions.items())

            for _ in xrange(self.n_iter):

                # Create new set of parameters to return
                params = dict()

                for k, v in items:

                    # Use numpy random choice to get random element from sequence
                    if self.is_list(v):
                        params[k] = np.random.choice(v)
                    # Call the function to get random number form distribution
                    elif hasattr(v, '__call__'):
                        params[k] = v()
                    else:
                        raise AttributeError("The key <%s> has neither a sequence nor a distribution attached" % k)
                        # end if

                # end for

                # Return iterator
                yield params

                # end for
        # end if
        return

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter

    def is_list(self, x):
        '''
        This function returns true if x is a list or sequence. This includes:
        - lists
        - numpy ndarray
        '''
        return isinstance(x, list) or isinstance(x, np.ndarray)


class ParameterManager(object):
    '''
    The parameter manager is initialized with a dictionary, keys should be the variable names as string, but can be any imutable object.
    Items are the desired values for the variable of arbitary type
    The manager can be called with the name of a variable and an optional default value to obtain the parameters.
    A typical example:

    # Define the parameters
    parameters = { 'alpha' : 0.1, 'gamma' : 0.99}

    # Create the paramter manager instance
    mngr = ParameterManager(parameters)

    # Default value is not used, because key 'gamma' is present
    gamma = mngr('gamma', default = 0.9)

    # This works, the key 'beta' is not present but a default value is given
    beta = mngr('beta', default = 0.75)

    # No Error: Key 'alpha' is in dict, the default value is not needed
    alpha = mngr('alpha')

    # KeyError: key 'delta' is not present, no default value is given
    delta = mngr('delta')
    '''

    def __init__(self, parameters):
        '''
        Create the ParamterManger from the given dictionary. See class description for more details
        '''
        if not isinstance(parameters, dict):
            raise TypeError("<%s> is not a dictionary" % parameters)
        # end

        self.parameters = parameters
        self.logger = logging.getLogger('ParameterManager')

        return

    def __call__(self, key, default=None):

        if key in self.parameters:
            return self.parameters[key]
        # end

        if default is not None:
            self.logger.warn("Using default value: <%s> = <%s>" % (key, default))
            return default
        # end

        raise KeyError("The key <%s> is not contained in parameters. No default value has been given")

    def __len__(self):
        return len(self.parameters)

    @staticmethod
    def get_example():
        '''
        This function returns an example for the parameter dictionary which is typical for my RL agents
        '''

        return {'alpha': 0.01,
                'gamma': 0.99,
                'eps_init': 1.0,
                'eps_fin': 1e-4,
                'eps_decay': 0.999,
                'eps_step': 1e-4,
                'beta': 0.75}


class ParameterTuner(object):
    '''
    This class uses either the parameter sampler or the parameter grid to optimize the specified
    hyper parameters.
    '''

    def __init__(self, callback_function, search_space, parameters):
        '''
        Creates an instance of the paramter tuner.

        callback_function:  function with following signature
                            def f( id, p ):

                              --- your code here to evaluate an agent ---

                              return score, p

                            This function gets called with an unique id and a parameter combination arguments.
                            This parameters can be parsed with the ParameterManager Class to obtain the values of each variable. See the ParameterManager documentation for more details.
                            The function has to return the score and the provided parameters as tuple.
                            It is absolute necessity to return the parameters. If the sampler is used, there is no way for the tuner
                            to determine which paramter combination belongs to the highest score.
                            Due to the parrallel execution, the file, in which the callback function is defined, may only contain definitions like :

                            class c(object):
                              pass

                            def f( *args, **kwargs):
                              pass

                            a = 42

                            Any other line, which creates instances or executes codes, has to be protected by:

                            if __name__ == '__main__':
                              --- your code here ---

                            Otherwise, each parrellel running worker thread of the tuner will execute this line resulting
                            in a recursive calling pattern.

        search_space:       dict of string to sequence OR dict string to scipy.stats distributions

                            The parameters to explore, e.g.:

                            parameters = {'a': [1, 2], 'b': [True, False]}

                            or

                            parameters = {'a':[1, 2], 'b': scipy.stats.expon()}

                            Every distribution works whichs implements the distribution interface of scipy (especially the rvs() method)
                            i.e. the distribution is a subclass of rv_continuous for continuous distribution or rv_discrete for discrete distributions

                            As soon as a single distribution is defined as legal range for a paramter the sampler is used. Otherwise
                            a grid search is performed. The use of the sampler can be forced.

        The following arguments are contained in a parameter dictionary >parameters<.
        Left column are the keys as string, right column are the values

        n_jobs:             int, dafault = 8
                            Number of jobs to run in parallel.
                            Using 'n_jobs=1' turns off parallel computing for debugging without changing the codepath

        pre_dispatch:       None,int or string, default = '2*n_jobs'
                            Controls the number of jobs that get dispatched during parallel execution.  Reducing this number
                            can be useful to avoid an explosion of memory consumption when more jobs
                            get dispatched than CPUs can process. This parameter can be:
                            - None, in which case all the jobs are immediately created and spawned.
                              Use this for lightweight and fast-running jobs, to avoid delays due to on-demand
                              spawning of the jobs
                            - An int, giving the exact number of total jobs that are spawned
                            - A string, giving an expression as a function of n_jobs, as in '2*n_jobs'

        n_iter:             int, default = 1000
                            Number of samples to generate if the paramter sampler is used.
                            This argument is ignored if the paramter grid is used

        use_sampler         boolean, default = False
                            If true, the paramter sampler is always used, otherwise the parameters define the method to generate combinations of parameters

        verbose:            int, default = 0
                            Controls the verbosity - the higher, the more messages.
                            Serves also as progess meter, verbose=5 results for example in:
                            [Parallel(n_jobs=2)]: Done  1 out of 10 | elapsed: 0.1s remaining: 0.9s
                            [Parallel(n_jobs=2)]: Done  3 out of 10 | elapsed: 0.2s remaining: 0.5s
                            [Parallel(n_jobs=2)]: Done  6 out of 10 | elapsed: 0.3s remaining: 0.2s
                            [Parallel(n_jobs=2)]: Done  9 out of 10 | elapsed: 0.5s remaining: 0.1s
                            [Parallel(n_jobs=2)]: Done 10 out of 10 | elapsed: 0.5s finished
        '''

        # Create the paramter manager for parsing
        mngr = ParameterManager(parameters)

        # Store the variables as class members
        self.callback_function = callback_function
        self.search_space = search_space

        self.n_jobs = mngr('n_jobs', default=8)
        self.pre_dispatch = mngr('pre_dispatch', default='2*n_jobs')
        self.verbose = mngr('verbose', default=0)
        self.n_iter = mngr('n_iter', default=100)

        # Check if distributions are present, for now, x is assumed to be a distribution if it is
        # callable and not a list
        is_distribution = lambda x: hasattr(x, '__call__') and not isinstance(x, list)
        has_distributions = np.any([is_distribution(x) for x in self.search_space.values()])
        use_sampler = mngr('use_sampler', default=False)

        # If distributions are present or if the usage of the sampler is enforced
        if has_distributions or use_sampler:
            self.generator = ParameterSampler(self.search_space, self.n_iter)
        else:
            self.generator = ParameterGrid(self.search_space)
        # end if

        logger.info("Parameter tuner is ready. %d tasks are defined. Using  the %s", len(self.generator),
                    "sampler" if has_distributions or use_sampler else "grid search")
        return

    def __call__(self, filename=None):
        '''
        Provides the () operator for this class, ParamterTuner() is the same as ParamterTuner.run()
        '''
        return self.run(filename)

    def run(self, filename=None):
        '''
        Runs the paramter tuning

        Collects all the resulting scores and returns the parameter combination with the highest score
        '''

        # Use contex manager to keep the same set of workers alive.
        # Only required if multiple successive Parallel exectutions are rquired
        # (or the execution of the workers is in a loop etc)
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch) as workers:
            # The workers take an iterator of the functions, which is used to execute
            # them ( based on n_jobs and pre_dispatch)
            # The line (delayed(test)(s) for s in sequence) produces a generator
            # object
            out = workers(delayed(self.callback_function)(id, params) for id, params in enumerate(self.generator))

        # end with

        # If desired, save all values to a file
        if filename is not None:
            self.save(out, filename)
            logger.info("Saving - completed")
        # end if

        # out is a list of (score,param) tuples
        # Get paramter by python's max
        # Key is a function that identifies how to retrieve the comparable
        # element from your data structure.
        score, p = max(out, key=lambda x: x[0])

        return p, score

    def save(self, output, filename):
        '''
        Writes the output of the workers to a text file
        Each row corresponds to one paramter combination of one worker
        The colums are sorted by alphabetical order of parameter names
        '''

        with file(filename, "w") as the_file:

            # Create header
            the_file.write("score ")
            for key in sorted(output[0][1].keys()):
                the_file.write("%s " % key)
            the_file.write("\n")

            # Loop over output
            for s, p in output:
                # Write the score
                the_file.write("%s " % s)

                # Write the params
                items = sorted(p.items(), key=lambda x: x[0])
                for _, value in items:
                    the_file.write("%s " % value)
                the_file.write("\n")
                # end for in out
        # end with file

        return


import time

def test_fun(id, p):
    '''
    Simple function to test the hyperparamter tuner
    with parallel execution.
    Important:
      two arguments: id and parameters
      returns: score AND parameters
    '''
    # Get value of paramter a
    a = p['a']

    # Parabola with maximum at (5,10)
    score = - (a - 5.0) ** 2.0 + 10.0

    time.sleep(2)

    return score, p


import ddpg3
def run_trial(id, p):
    lr = p['lr']
    ddpg3_instance = ddpg3.ddpg(environment='MountainCarContinuous-v0',
                                learning_rates= lr,
                                enable_plotting= False)

    score = ddpg3_instance.main()


    return score, p


# Example for tuner
if __name__ == '__main__':
    p = {'n_jobs': 1,
         'pre_dispatch': '2*n_jobs',
         'n_iter': 10,
         'use_sampler': False,
         'verbose': 5}

    lr = np.linspace(1e-5, 1e-2, num=5)
    lr = zip(lr*0.1,lr)  # the order is actor_lr critic_lr, the critic needs to be faster
    print(lr)
    # search_space = {'lr': lr}
    # search_space = {'a': lambda: np.random.uniform(0,10), 'b': lambda: np.random.uniform(0,1)}
    search_space = {'a': range(10)}


    tuner = ParameterTuner(test_fun, search_space, p)
    result = tuner('hyperparameters.txt')

    logger.info("Result %s", result)
