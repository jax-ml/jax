from functools import partial
from typing import Any, Callable, Mapping, Optional, Tuple

from ._bfgs import minimize_bfgs
from typing import NamedTuple
import jax.numpy as jnp


class OptimizeResults(NamedTuple):
  """Object holding optimization results.

  Parameters:
    x: final solution.
    success: True if optimization succeeded.
    status: solver specific return code. 0 means nominal.
    message: solver specific message.
    fun: final function value.
    jac: final jacobian array.
    hess_inv: final inverse Hessian estimate.
    nfev: number of funcation calls used.
    njev: number of gradient evaluations.
    nit: number of iterations of the optimization algorithm.
  """
  x: jnp.ndarray
  success: bool
  status: int
  message: str
  fun: jnp.ndarray
  jac: jnp.ndarray
  hess_inv: jnp.ndarray
  nfev: int
  njev: int
  nit: int


def minimize(
    fun: Callable,
    x0: jnp.ndarray,
    args: Tuple = (),
    *,
    method: Optional[str] = None,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> OptimizeResults:
  """Minimization of scalar function of one or more variables.

  This API for this function matches SciPy with some minor deviations:
  - Gradients of ``fun`` are calculated automatically using JAX's autodiff
    support when required.
  - The ``method`` argument is required. You must specify a solver.
  - Various optional arguments in the SciPy interface have not yet been
    implemented.
  - Optimization results may differ from SciPy due to differences in the line
    search implementation.

  ``minimize`` supports ``jit`` compilation. It does not yet support
   differentiation or arguments in the form of multi-dimensional arrays, but
   support for both is planned.

  Args:
    fun: the objective function to be minimized, ``fun(x, *args) -> float``,
      where ``x`` is an 1-D array with shape ``(n,)`` and ``args`` is a tuple
      of the fixed parameters needed to completely specify the function.
      ``fun`` must support differentiation.
    x0: initial guess. Array of real elements of size ``(n,)``, where 'n' is
      the number of independent variables.
    args: extra arguments passed to the objective function.
    method: solver type. Currently only "BFGS" is supported.
    tol: tolerance for termination. For detailed control, use solver-specific
      options.
    options: a dictionary of solver options. All methods accept the following
      generic options:
        maxiter : int
            Maximum number of iterations to perform. Depending on the
            method each iteration may use several function evaluations.

  Returns: OptimizeResults object.
  """
  if options is None:
    options = {}

  fun_with_args = partial(fun, *args)

  if method.lower() == 'bfgs':
    results = minimize_bfgs(fun_with_args, x0, **options)
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
