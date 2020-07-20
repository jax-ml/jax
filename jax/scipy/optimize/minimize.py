from .bfgs_minimize import minimize_bfgs
from typing import NamedTuple
import jax.numpy as jnp


class OptimizeResults(NamedTuple):
  x: jnp.ndarray  # Final variable
  success: bool  # Wheter optimization converged (and there were no other failures, e.g. line search failures)
  status: int  # Solver specific return code. 0 means nominal
  message: str  # Soler specific message
  fun: jnp.ndarray  # Final function value
  jac: jnp.ndarray  # Final jacobian array
  hess_inv: jnp.ndarray  # Final inverse Hessian estimate
  nfev: int  # Number of funcation calls used
  njev: int  # Number of gradient evaluations
  nit: int  # Number of iterations of the optimization algorithm


def minimize(fun, x0, *, args=(), method=None, tol=None, options=None):
  """
  Interface to scalar function minimisation.

  This implementation is jittable so long as `fun` is.
  Args:
      fun: jax function
      x0: initial guess, currently only single flat arrays supported.
      args: tuple, optional
          Extra arguments to pass to func as func(x,*args)
      method: Available methods: ['BFGS']
      tol: Tolerance for termination. For detailed control, use solver-specific options.
      options: A dictionary of solver options. All methods accept the following generic options:
          maxiter : int
              Maximum number of iterations to perform. Depending on the
              method each iteration may use several function evaluations.

  Returns: OptimizeResults

  """
  if method.lower() == 'bfgs':
    results = minimize_bfgs(fun, x0, args=args, **options)
    message = ("status meaning: 0=converged, 1=max BFGS iters reached, "
               "3=zoom failed, 4=saddle point reached, "
               "5=max line search iters reached, -1=undefined")
    return OptimizeResults(x=results.x_k,
                           success=(results.converged) & (~results.failed),
                           status=results.status,
                           message=message,
                           fun=results.f_k,
                           jac=results.g_k,
                           hess_inv=results.H_k,
                           nfev=results.nfev,
                           njev=results.ngev,
                           nit=results.k)

  raise ValueError("Method {} not recognized".format(method))
