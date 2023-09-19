``jax.random`` module
=====================

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
    key
    key_data
    wrap_key_data
    ball
    bernoulli
    beta
    bits
    categorical
    cauchy
    chisquare
    choice
    dirichlet
    double_sided_maxwell
    exponential
    f
    fold_in
    gamma
    generalized_normal
    geometric
    gumbel
    laplace
    loggamma
    logistic
    lognormal
    maxwell
    multivariate_normal
    normal
    orthogonal
    pareto
    permutation
    poisson
    rademacher
    randint
    rayleigh
    shuffle
    split
    t
    triangular
    truncated_normal
    uniform
    wald
    weibull_min

