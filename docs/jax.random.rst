jax.random package
==================

.. automodule:: jax.random

List of Available Functions
---------------------------

.. Generate the list below as follows:
   >>> from jax import random
   >>> fns = (x for x in sorted(dir(random)) if x != 'threefry_2x32')
   >>> fns = (x for x in fns if callable(getattr(random, x)))
   >>> print('\n'.join('    ' + x for x in fns))  # doctest: +SKIP

.. autosummary::
  :toctree: _autosummary
      
    PRNGKey
    bernoulli
    beta
    categorical
    cauchy
    choice
    dirichlet
    double_sided_maxwell
    exponential
    fold_in
    gamma
    gumbel
    laplace
    logistic
    maxwell
    multivariate_normal
    normal
    pareto
    permutation
    poisson
    rademacher
    randint
    shuffle
    split
    t
    truncated_normal
    uniform
    weibull_min

