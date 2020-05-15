from .bfgs_minimize import bfgs_minimize


def minimize(fun, x0, *, method=None, tol=None, options=None):
    """
    Interface to scalar function minimisation.

    This implementation is jittable so long as `fun` is.
    Args:
        fun: jax function
        x0: initial guess, currently only single flat arrays supported.
        method: Available methods: ['BFGS']
        tol: Tolerance for termination. For detailed control, use solver-specific options.
        options: A dictionary of solver options. All methods accept the following generic options:
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.

    Returns:

    """
    if method.lower() == 'bfgs':
        return bfgs_minimize(fun, x0, options=options)
    raise ValueError("Method {} not recognised".format(method))
