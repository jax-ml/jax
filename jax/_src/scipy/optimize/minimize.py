# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

import jax
from typing import NamedTuple


class OptimizeResults(NamedTuple):
  """Object holding optimization results.

  Parameters:
    x: final solution.
    success: ``True`` if optimization succeeded.
    status: integer solver specific return code. 0 means converged (nominal),
      1=max BFGS iters reached, 3=zoom failed, 4=saddle point reached,
      5=max line search iters reached, -1=undefined
    fun: final function value.
    jac: final jacobian array.
    hess_inv: final inverse Hessian estimate.
    nfev: integer number of function calls used.
    njev: integer number of gradient evaluations.
    nit: integer number of iterations of the optimization algorithm.
  """
  x: jax.Array
  success: Union[bool, jax.Array]
  status: Union[int, jax.Array]
  fun: jax.Array
  jac: jax.Array
  hess_inv: Optional[jax.Array]
  nfev: Union[int, jax.Array]
  njev: Union[int, jax.Array]
  nit: Union[int, jax.Array]


def minimize(
    fun: Callable,
    x0: jax.Array,
    args: tuple = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> OptimizeResults:
  """Minimization of scalar function of one or more variables.

  This is a scipy-like wrapper around functionality available in the
  jaxopt_ package. The API for this function matches SciPy with some
  minor deviations:

  - Gradients of ``fun`` are calculated automatically using JAX's autodiff
    support when required.
  - The ``method`` argument is required. You must specify a solver.
  - Various optional arguments in the SciPy interface have not yet been
    implemented.
  - Optimization results may differ from SciPy due to differences in the line
    search implementation.

  Args:
    fun: the objective function to be minimized, ``fun(x, *args) -> float``,
      where ``x`` is a 1-D array with shape ``(n,)`` and ``args`` is a tuple
      of the fixed parameters needed to completely specify the function.
      ``fun`` must support differentiation.
    x0: initial guess. Array of real elements of size ``(n,)``, where ``n`` is
      the number of independent variables.
    args: extra arguments passed to the objective function.
    method: solver type. Currently only ``"BFGS"`` is supported.
    tol: tolerance for termination. For detailed control, use solver-specific
      options.
    options: a dictionary of solver options. All methods accept the following
      generic options:

      - maxiter (int): Maximum number of iterations to perform. Depending on the
        method each iteration may use several function evaluations.

  Returns:
    An :class:`OptimizeResults` object.

  .. _jaxopt: https://jaxopt.github.io/
  """
  try:
    import jaxopt
  except ImportError as err:
    raise RuntimeError("jaxopt package must be installed to use jax.scipy.optimize.minimize") from err

  if not isinstance(args, tuple):
    raise TypeError(f"args {args} must be a tuple")

  def fun_with_args(x):
    return fun(x, *args)

  if options is None:
    options = {}

  if tol is None:
    if (gtol := options.pop('gtol', None)) is not None:
      tol = gtol
    else:
      tol = 1E-6
  if (maxiter := options.pop('maxiter', None)) is None:
    maxiter = 500

  if options:
    raise ValueError(f"Unrecognized options: {list(options.keys())}")

  if method.lower() == 'bfgs':
    solver = jaxopt.BFGS(fun=fun_with_args, tol=tol, maxiter=maxiter, maxls=30)
  elif method.lower() == 'l-bfgs-experimental-do-not-rely-on-this':
    solver = jaxopt.LBFGS(fun=fun_with_args, tol=tol, maxiter=maxiter, maxls=30)
  else:
    raise ValueError(f"Method {method} not recognized")

  params, state = solver.run(x0)
  return OptimizeResults(
    x=params,
    success=state.error <= solver.tol,
    status=0,  # No status flag in jaxopt.
    fun=state.value,
    jac=state.grad,
    hess_inv=getattr(state, 'H', None),
    nfev=state.num_fun_eval,
    njev=state.num_grad_eval,
    nit=state.iter_num,
  )
