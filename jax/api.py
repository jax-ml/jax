# coding=utf-8
# Copyright 2018 Google LLC
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

"""
JAX user-facing transformations and utilities.

The transformations here mostly wrap internal transformations, providing
convenience flags to control behavior and handling Python containers of
arguments and outputs. The Python containers handled are pytrees (see
tree_util.py), which include nested tuples/lists/dicts, where the leaves are
arrays.
"""

# flake8: noqa: F401
import collections
import functools
import inspect
import itertools as it
import threading
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as onp
from contextlib import contextmanager

from . import core
from . import linear_util as lu
from . import ad_util
from . import dtypes
from .core import eval_jaxpr
from .api_util import (wraps, flatten_fun, apply_flat_fun, flatten_fun_nokwargs,
                       flatten_fun_nokwargs2, argnums_partial, flatten_axes,
                       donation_vector, rebase_donate_argnums)
from .tree_util import (tree_map, tree_flatten, tree_unflatten, tree_structure,
                        tree_transpose, tree_leaves, tree_multimap,
                        treedef_is_leaf)
from .util import (unzip2, curry, partial, safe_map, safe_zip, prod,
                   split_list, extend_name_stack, wrap_name)
from .lib import xla_bridge as xb
from .lib import xla_client as xc
# Unused imports to be exported
from .lib.xla_bridge import (device_count, local_device_count, devices, local_devices,
                             host_id, host_ids, host_count)
from .abstract_arrays import ConcreteArray, ShapedArray, raise_to_shaped
from .interpreters import partial_eval as pe
from .interpreters import xla
from .interpreters import pxla
from .interpreters import ad
from .interpreters import batching
from .interpreters import parallel
from .interpreters import masking
from .interpreters import invertible_ad as iad
from .interpreters.invertible_ad import custom_ivjp
from .custom_derivatives import custom_jvp, custom_vjp
from .config import flags, config, bool_env

AxisName = Any

map = safe_map
zip = safe_zip

FLAGS = flags.FLAGS
flags.DEFINE_bool("jax_disable_jit",
                  bool_env("JAX_DISABLE_JIT", False),
                  "Disable JIT compilation and just call original Python.")


def _check_callable(fun):
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if inspect.isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

class _ThreadLocalState(threading.local):
  def __init__(self):
    self.jit_is_disabled = False

_thread_local_state = _ThreadLocalState()

def jit(fun: Callable, static_argnums: Union[int, Iterable[int]] = (),
        device=None, backend: Optional[str] = None,
        donate_argnums: Union[int, Iterable[int]] = ()) -> Callable:
  """Sets up ``fun`` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    static_argnums: An int or collection of ints specifying which positional
      arguments to treat as static (compile-time constant). Operations that only
      depend on static arguments will be constant-folded. Calling the jitted
      function with different values for these constants will trigger
      recompilation. If the jitted function is called with fewer positional
      arguments than indicated by ``static_argnums`` then an error is raised.
      Arguments that are not arrays or containers thereof must be marked as
      static. Defaults to ().
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited from
      XLA's DeviceAssignment logic and is usually to use ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perfom a computation, for
      example recycling one of your input buffers to store a result. You should
      not re-use buffers that you donate to a computation, JAX will raise an
      error if you try to.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

  In the following example, ``selu`` can be compiled into a single fused kernel
  by XLA:

  >>> import jax
  >>>
  >>> @jax.jit
  ... def selu(x, alpha=1.67, lmbda=1.05):
  ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
  >>>
  >>> key = jax.random.PRNGKey(0)
  >>> x = jax.random.normal(key, (10,))
  >>> print(selu(x))  # doctest: +SKIP
  [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
   -0.85743 -0.78232  0.76827  0.59566 ]
  """
  _check_callable(fun)
  static_argnums = _ensure_tuple(static_argnums)
  donate_argnums = _ensure_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  @wraps(fun)
  def f_jitted(*args, **kwargs):
    if _jit_is_disabled():
      return fun(*args, **kwargs)
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      msg = ("jitted function has static_argnums={}, donate_argnums={} but "
             "was called with only {} positional arguments.")
      raise ValueError(msg.format(static_argnums, donate_argnums, len(args)))
    f = lu.wrap_init(fun)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f, dyn_args = argnums_partial(f, dyn_argnums, args)
    else:
      dyn_args = args
    args_flat, in_tree = tree_flatten((dyn_args, kwargs))
    donated_invars = donation_vector(donate_argnums, dyn_args, kwargs)
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)
    out = xla.xla_call(flat_fun, *args_flat, device=device, backend=backend,
                       name=flat_fun.__name__, donated_invars=donated_invars)
    return tree_unflatten(out_tree(), out)

  return f_jitted

@contextmanager
def disable_jit():
  """Context manager that disables :py:func:`jit` behavior under its dynamic context.

  For debugging it is useful to have a mechanism that disables :py:func:`jit`
  everywhere in a dynamic context.

  Values that have a data dependence on the arguments to a jitted function are
  traced and abstracted. For example, an abstract value may be a
  :py:class:`ShapedArray` instance, representing the set of all possible arrays
  with a given shape and dtype, but not representing one concrete array with
  specific values. You might notice those if you use a benign side-effecting
  operation in a jitted function, like a print:

  >>> import jax
  >>>
  >>> @jax.jit
  ... def f(x):
  ...   y = x * 2
  ...   print("Value of y is", y)
  ...   return y + 3
  ...
  >>> print(f(jax.numpy.array([1, 2, 3])))
  Value of y is Traced<ShapedArray(int32[3]):JaxprTrace(level=-1/1)>
  [5 7 9]

  Here ``y`` has been abstracted by :py:func:`jit` to a :py:class:`ShapedArray`,
  which represents an array with a fixed shape and type but an arbitrary value.
  The value of ``y`` is also traced. If we want to see a concrete value while
  debugging, and avoid the tracer too, we can use the :py:func:`disable_jit`
  context manager:

  >>> import jax.numpy as np
  >>>
  >>> with jax.disable_jit():
  ...   print(f(np.array([1, 2, 3])))
  ...
  Value of y is [2 4 6]
  [5 7 9]
  """
  try:
    prev_val = _thread_local_state.jit_is_disabled
    _thread_local_state.jit_is_disabled = True
    yield
  finally:
    _thread_local_state.jit_is_disabled = prev_val

def _jit_is_disabled():
  return _thread_local_state.jit_is_disabled or config.read('jax_disable_jit')


def xla_computation(fun: Callable,
                    static_argnums: Union[int, Iterable[int]] = (),
                    axis_env: Optional[Sequence[Tuple[AxisName, int]]] = None,
                    backend: Optional[str] = None,
                    tuple_args: bool = False,
                    instantiate_const_outputs: bool = True) -> Callable:
  """Creates a function that produces its XLA computation given example args.

  Args:
    fun: Function from which to form XLA computations.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`. See the examples below.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    tuple_args: Optional bool, defaults to ``False``. If ``True``, the resulting
      XLA computation will have a single tuple argument that is unpacked into
      the specified function arguments.
    instantiate_const_outputs: Optional bool, defaults to ``True``. If
      ``False``, then :py:func:`xla_computation` does not instantiate
      constant-valued outputs in the XLA computation, and so the result is
      closer to the computation that :py:func:`jax.jit` produces and may be more
      useful for studying :py:func:`jit` behavior. If ``True``, then
      constant-valued outputs are instantiated in the XLA computation, which may
      be more useful for staging computations out of JAX entirely.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns a
    built XLA Computation (see xla_client.py), from which representations of the
    unoptimized XLA HLO computation can be extracted using methods like
    ``as_hlo_text``, ``as_serialized_hlo_module_proto``, and
    ``as_hlo_dot_graph``.

  For example:

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> c = jax.xla_computation(f)(3.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule xla_computation_f.6
  <BLANKLINE>
  ENTRY xla_computation_f.6 {
    constant.2 = pred[] constant(false)
    parameter.1 = f32[] parameter(0)
    cosine.3 = f32[] cosine(parameter.1)
    sine.4 = f32[] sine(cosine.3)
    ROOT tuple.5 = (f32[]) tuple(sine.4)
  }
  <BLANKLINE>
  <BLANKLINE>


  Here's an example that involves a parallel collective and axis name:

  >>> def f(x): return x - jax.lax.psum(x, 'i')
  >>> c = jax.xla_computation(f, axis_env=[('i', 4)])(2)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation.9
  primitive_computation.3 {
    parameter.4 = s32[] parameter(0)
    parameter.5 = s32[] parameter(1)
    ROOT add.6 = s32[] add(parameter.4, parameter.5)
  }
  ENTRY jaxpr_computation.9 {
    tuple.1 = () tuple()
    parameter.2 = s32[] parameter(0)
    all-reduce.7 = s32[] all-reduce(parameter.2), replica_groups={{0,1,2,3}}, to_apply=primitive_computation.3
    ROOT subtract.8 = s32[] subtract(parameter.2, all-reduce.7)
  }
  <BLANKLINE>
  <BLANKLINE>

  Notice the ``replica_groups`` that were generated. Here's an example that
  generates more interesting ``replica_groups``:

  >>> from jax import lax
  >>> def g(x):
  ...   rowsum = lax.psum(x, 'i')
  ...   colsum = lax.psum(x, 'j')
  ...   allsum = lax.psum(x, ('i', 'j'))
  ...   return rowsum, colsum, allsum
  ...
  >>> axis_env = [('i', 4), ('j', 2)]
  >>> c = xla_computation(g, axis_env=axis_env)(5.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation__1.19
  [removed uninteresting text here]
  ENTRY jaxpr_computation__1.19 {
    tuple.1 = () tuple()
    parameter.2 = f32[] parameter(0)
    all-reduce.7 = f32[] all-reduce(parameter.2), replica_groups={{0,2,4,6},{1,3,5,7}}, to_apply=primitive_computation__1.3
    all-reduce.12 = f32[] all-reduce(parameter.2), replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=primitive_computation__1.8
    all-reduce.17 = f32[] all-reduce(parameter.2), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=primitive_computation__1.13
    ROOT tuple.18 = (f32[], f32[], f32[]) tuple(all-reduce.7, all-reduce.12, all-reduce.17)
  }
  """
  _check_callable(fun)
  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)
  fun_name = getattr(fun, '__name__', 'unknown')

  def make_axis_env(nreps):
    if axis_env is None:
      return xla.AxisEnv(nreps)
    else:
      nreps = nreps * prod(size for name, size in axis_env)
      names, sizes = zip(*axis_env)
      return xla.AxisEnv(nreps, names, sizes)

  def abstractify(x):
    return ShapedArray(onp.shape(x), dtypes.result_type(x))

  @wraps(fun)
  def computation_maker(*args, **kwargs):
    wrapped = lu.wrap_init(fun)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      wrapped, _ = argnums_partial(wrapped, dyn_argnums, args)
    jax_args, in_tree = tree_flatten((args, kwargs))
    jaxtree_fun, out_tree = flatten_fun(wrapped, in_tree)
    avals = map(abstractify, jax_args)
    pvals = [pe.PartialVal.unknown(aval) for aval in avals]
    jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals,
                                         instantiate=instantiate_const_outputs,
                                         stage_out=True)
    jaxpr, _ = xla.apply_outfeed_rewriter(jaxpr)
    axis_env_ = make_axis_env(xla.jaxpr_replicas(jaxpr))
    c = xb.make_computation_builder('xla_computation_{}'.format(fun_name))
    xla_consts = map(partial(xb.constant, c), consts)
    xla_args = xla._xla_callable_args(c, avals, tuple_args)
    outs = xla.jaxpr_subcomp(
        c, jaxpr, backend, axis_env_, xla_consts,
        extend_name_stack(wrap_name(fun_name, 'xla_computation')), *xla_args)
    return c.build(xc.ops.Tuple(c, outs))
  return computation_maker

def grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
         has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """Creates a function which evaluates the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers.
      Argument arrays in the positions specified by ``argnums`` must be of
      inexact (i.e., floating-point or complex) type. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.

  For example:

  >>> import jax
  >>>
  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux,
                                    holomorphic=holomorphic)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def grad_f(*args, **kwargs):
    _, g = value_and_grad_f(*args, **kwargs)
    return g

  @wraps(fun, docstr=docstr, argnums=argnums)
  def grad_f_aux(*args, **kwargs):
    (_, aux), g = value_and_grad_f(*args, **kwargs)
    return g, aux

  return grad_f_aux if has_aux else grad_f

def value_and_grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False
                   ) -> Callable[..., Tuple[Any, Any]]:
  """Create a function which evaluates both ``fun`` and the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.

  Returns:
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a sequence of integers, the gradient is a tuple of values with the same
    shapes and types as the corresponding arguments.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  _check_callable(fun)

  @wraps(fun, docstr=docstr, argnums=argnums)
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if type(argnums) is int else max(argnums)
    if max_argnum >= len(args):
      msg = ("differentiating with respect to argnums={} requires at least "
             "{} positional arguments to be passed by the caller, but got only "
             "{} positional arguments.")
      raise TypeError(msg.format(argnums, max_argnum + 1, len(args)))

    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_grad, holomorphic), dyn_args)
    if not has_aux:
      ans, vjp_py = _vjp(f_partial, *dyn_args)
    else:
      ans, vjp_py, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    _check_scalar(ans)
    dtype = dtypes.result_type(ans)
    tree_map(partial(_check_output_dtype_grad, holomorphic), ans)
    g = vjp_py(onp.ones((), dtype=dtype))
    g = g[0] if isinstance(argnums, int) else g
    if not has_aux:
      return ans, g
    else:
      return (ans, aux), g

  return value_and_grad_f

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.get_aval(x)
  except TypeError as e:
    raise TypeError(msg("was {}".format(x))) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        raise TypeError(msg("had shape: {}".format(aval.shape)))
    else:
      raise TypeError(msg("had abstract value {}".format(aval)))

def _check_input_dtype_revderiv(name, holomorphic, x):
  _check_arg(x)
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, onp.complexfloating):
      msg = (f"{name} with holomorphic=True requires inputs with complex dtype, "
             f"but got {aval.dtype.name}.")
      raise TypeError(msg)
  elif not (dtypes.issubdtype(aval.dtype, onp.floating) or
            dtypes.issubdtype(aval.dtype, onp.complexfloating)):
    msg = (f"{name} requires real- or complex-valued inputs (input dtype that "
           "is a sub-dtype of np.floating or np.complexfloating), "
           f"but got {aval.dtype.name}. ")
    raise TypeError(msg)
_check_input_dtype_grad = partial(_check_input_dtype_revderiv, "grad")

def _check_output_dtype_revderiv(name, holomorphic, x):
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, onp.complexfloating):
      msg = (f"{name} with holomorphic=True requires outputs with complex dtype, "
             f"but got {aval.dtype.name}.")
      raise TypeError(msg)
  elif not dtypes.issubdtype(aval.dtype, onp.floating):
    msg = (f"{name} requires real-valued outputs (output dtype that is "
           f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
           "For holomorphic differentiation, pass holomorphic=True. "
           "For differentiation of non-holomorphic functions involving complex "
           "outputs, use jax.vjp directly.")
    raise TypeError(msg)
_check_output_dtype_grad = partial(_check_output_dtype_revderiv, "grad")


def jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation.

  >>> import jax
  >>> import jax.numpy as np
  >>>
  >>> def f(x):
  ...   return jax.numpy.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(np.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd = partial(_jvp, f_partial, dyn_args)
    y, jac = vmap(pushfwd, out_axes=(None, batching.last))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    return tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)

  return jacfun

def _check_input_dtype_jacfwd(holomorphic, x):
  _check_arg(x)
  aval = core.get_aval(x)
  if holomorphic:
    if not (dtypes.issubdtype(aval.dtype, onp.complexfloating) and
            not dtypes.issubdtype(aval.dtype, onp.floating)):
      msg = ("jacfwd with holomorphic=True requires inputs with complex dtype, "
             f"but got {aval.dtype.name}.")
      raise TypeError(msg)
  elif not dtypes.issubdtype(aval.dtype, onp.floating):
    msg = ("jacfwd requires real-valued inputs (input dtype that is "
           f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
           "For holomorphic differentiation, pass holomorphic=True. "
           "For differentiation of non-holomorphic functions involving complex "
           "inputs, use jax.jvp directly.")
    raise TypeError(msg)

def _check_output_dtype_jacfwd(holomorphic, x):
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, onp.complexfloating):
      msg = ("jacfwd with holomorphic=True requires outputs with complex dtype, "
             f"but got {aval.dtype.name}.")
      raise TypeError(msg)


def jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation.

  >>> import jax
  >>> import jax.numpy as np
  >>>
  >>> def f(x):
  ...   return jax.numpy.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(np.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    return tree_transpose(tree_structure(example_args), tree_structure(y), jac)

  return jacfun
jacobian = jacrev

_check_input_dtype_jacrev = partial(_check_input_dtype_revderiv, "jacrev")
_check_output_dtype_jacrev = partial(_check_output_dtype_revderiv, "jacrev")


def hessian(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
            holomorphic: bool = False) -> Callable:
  """Hessian of ``fun`` as a dense array.

  Args:
    fun: Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Hessian of
    ``fun``.

  >>> import jax
  >>>
  >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
  >>> print(jax.hessian(g)(jax.numpy.array([1., 2.])))
  [[   6.   -2.]
   [  -2. -480.]]

  :py:func:`hessian` is a generalization of the usual definition of the Hessian
  that supports nested Python containers (i.e. pytrees) as inputs and outputs.
  The tree structure of ``jax.hessian(fun)(x)`` is given by forming a tree
  product of the structure of ``fun(x)`` with a tree product of two copies of
  the structure of ``x``. A tree product of two tree structures is formed by
  replacing each leaf of the first tree with a copy of the second. For example:

  >>> import jax.numpy as jnp
  >>> f = lambda dct: {"c": jnp.power(dct["a"], dct["b"])}
  >>> print(jax.hessian(f)({"a": jnp.arange(2.) + 1., "b": jnp.arange(2.) + 2.}))
  {'c': {'a': {'a': DeviceArray([[[ 2.,  0.], [ 0.,  0.]],
                                 [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
               'b': DeviceArray([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                                 [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
         'b': {'a': DeviceArray([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                                 [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
               'b': DeviceArray([[[0.      , 0.      ], [0.      , 0.      ]],
                                [[0.      , 0.      ], [0.      , 3.843624]]], dtype=float32)}}}

  Thus each leaf in the tree structure of ``jax.hessian(fun)(x)`` corresponds to
  a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
  ``jax.hessian(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
  shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
  shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
  then the Hessian leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
  in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
  the block structure of the Hessian, with blocks determined by the input and
  output pytrees.

  In particular, an array is produced (with no pytrees involved) when the
  function input ``x`` and output ``fun(x)`` are each a single array, as in the
  ``g`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
  has shape ``(in1, in2, ...)`` then ``jax.hessian(fun)(x)`` has shape
  ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
  1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
  """
  return jacfwd(jacrev(fun, argnums, holomorphic), argnums, holomorphic)

def _std_basis(pytree):
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(onp.size, leaves))
  # TODO(mattjj): use a symbolic identity matrix here
  dtype = dtypes.result_type(*leaves)
  flat_basis = onp.eye(ndim, dtype=dtype)
  return _unravel_array_into_pytree(pytree, 1, flat_basis)

def _unravel_array_into_pytree(pytree, axis, arr):
  leaves, treedef = tree_flatten(pytree)
  axis = axis % arr.ndim
  shapes = [arr.shape[:axis] + onp.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, onp.cumsum(map(onp.size, leaves[:-1])), axis)
  reshaped_parts = [onp.reshape(x, shape) for x, shape in zip(parts, shapes)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, onp.ndarray):
    return onp.split(x, indices, axis)
  else:
    return x.split(indices, axis)

def _dtype(x):
  return dtypes.canonicalize_dtype(dtypes.result_type(x))


def vmap(fun: Callable, in_axes=0, out_axes=0) -> Callable:
  """Vectorizing map. Creates a function which maps ``fun`` over argument axes.

  Args:
    fun: Function to be mapped over additional axes.
    in_axes: A nonnegative integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof specifying which input array axes to map over.
      If each positional argument to ``fun`` is an array, then ``in_axes`` can
      be a nonnegative integer, a None, or a tuple of integers and Nones with
      length equal to the number of positional arguments to ``fun``. An integer
      or None indicates which array axis to map over for all arguments (with
      None indicating not to map any axis), and a tuple indicates which axis to
      map for each corresponding positional argument. If the positional
      arguments to ``fun`` are container types, the corresponding element of
      ``in_axes`` can itself be a matching container, so that distinct array
      axes can be mapped for different container elements. ``in_axes`` must be a
      container tree prefix of the positional argument tuple passed to ``fun``.

      At least one positional argument must have ``in_axes`` not None. The sizes
      of the mapped input axes for all mapped positional arguments must all
      be equal.

    out_axes: A nonnegative integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof indicating where the mapped axis should appear
      in the output. All outputs with a mapped axis must have a non-None
      ``out_axes`` specification.

  Returns:
    Batched/vectorized version of ``fun`` with arguments that correspond to
    those of ``fun``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``fun``, but
    with extra array axes at positions indicated by ``out_axes``.

  For example, we can implement a matrix-matrix product using a vector dot
  product:

  >>> import jax.numpy as np
  >>>
  >>> vv = lambda x, y: np.vdot(x, y)  #  ([a], [a]) -> []
  >>> mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
  >>> mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

  Here we use ``[a,b]`` to indicate an array with shape (a,b). Here are some
  variants:

  >>> mv1 = vmap(vv, (0, 0), 0)   #  ([b,a], [b,a]) -> [b]        (b is the mapped axis)
  >>> mv2 = vmap(vv, (0, 1), 0)   #  ([b,a], [a,b]) -> [b]        (b is the mapped axis)
  >>> mm2 = vmap(mv2, (1, 1), 0)  #  ([b,c,a], [a,c,b]) -> [c,b]  (c is the mapped axis)

  Here's an example of using container types in ``in_axes`` to specify which
  axes of the container elements to map over:

  >>> A, B, C, D = 2, 3, 4, 5
  >>> x = np.ones((A, B))
  >>> y = np.ones((B, C))
  >>> z = np.ones((C, D))
  >>> def foo(tree_arg):
  ...   x, (y, z) = tree_arg
  ...   return np.dot(x, np.dot(y, z))
  >>> tree = (x, (y, z))
  >>> print(foo(tree))
  [[12. 12. 12. 12. 12.]
   [12. 12. 12. 12. 12.]]
  >>> from jax import vmap
  >>> K = 6  # batch size
  >>> x = np.ones((K, A, B))  # batch axis in different locations
  >>> y = np.ones((B, K, C))
  >>> z = np.ones((C, D, K))
  >>> tree = (x, (y, z))
  >>> vfoo = vmap(foo, in_axes=((0, (1, 2)),))
  >>> print(vfoo(tree).shape)
  (6, 2, 5)

  Here's another example using container types in ``in_axes``, this time a
  dictionary, to specify the elements of the container to map over:

  >>> dct = {'a': 0., 'b': np.arange(5.)}
  >>> x = 1.
  >>> def foo(dct, x):
  ...  return dct['a'] + dct['b'] + x
  >>> out = vmap(foo, in_axes=({'a': None, 'b': 0}, None))(dct, x)
  >>> print(out)
  [1. 2. 3. 4. 5.]

  The results of a vectorized function can be mapped or unmapped.
  For example, the function below returns a pair with the first
  element mapped and the second unmapped. Only for unmapped results
  we can specify ``out_axes`` to be ``None`` (to keep it unmapped).

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(np.arange(2.), 4.))
  (DeviceArray([4., 5.], dtype=float32), 8.0)

  If the ``out_axes`` is specified for an unmapped result, the result is broadcast
  across the mapped axis:

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(np.arange(2.), 4.))
  (DeviceArray([4., 5.], dtype=float32), DeviceArray([8., 8.], dtype=float32))

  If the ``out_axes`` is specified for a mapped result, the result is
  transposed accordingly.
  """
  _check_callable(fun)
  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  if isinstance(in_axes, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_axes = tuple(in_axes)

  if not all(isinstance(l, (type(None), int, batching._Last))
             for l in tree_leaves(in_axes)):
    msg = ("vmap in_axes must be an int, None, or (nested) container with "
           "those types as leaves, but got {}.")
    raise TypeError(msg.format(in_axes))
  if not all(isinstance(l, (type(None), int, batching._Last))
             for l in tree_leaves(out_axes)):
    msg = ("vmap out_axes must be an int, None, or (nested) container with "
           "those types as leaves, but got {}.")
    raise TypeError(msg.format(out_axes))

  @wraps(fun, docstr=docstr)
  def batched_fun(*args):
    args_flat, in_tree  = tree_flatten(args)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes)
    _ = _mapped_axis_size(in_tree, args_flat, in_axes_flat, "vmap")
    out_flat = batching.batch(flat_fun, args_flat, in_axes_flat,
                              lambda: flatten_axes("vmap out_axes", out_tree(),
                                                   out_axes))
    return tree_unflatten(out_tree(), out_flat)

  return batched_fun

def _get_axis_size(name: str, i:int, shape: Tuple[int, ...], axis: int):
  try:
    return shape[axis]
  except (IndexError, TypeError) as e:
    raise ValueError(f"{name} got arg {i} of rank {len(shape)} "
                     f"but axis to be mapped {axis}") from e

def _mapped_axis_size(tree, vals, dims, name):
  mapped_axis_sizes = {_get_axis_size(name, i, onp.shape(x), d)
                       for i, (x, d) in enumerate(zip(vals, dims))
                       if d is not None}
  try:
    size, = mapped_axis_sizes
    return size
  except ValueError as e:
    if not mapped_axis_sizes:
      raise ValueError(f"{name} must have at least one non-None value in in_axes") from e
    msg = "{} got inconsistent sizes for array axes to be mapped:\n".format(name) + "{}"
    # we switch the error message based on whether args is a tuple of arrays,
    # in which case we can produce an error message based on argument indices,
    # or if it has nested containers.
    # TODO(mattjj,phawkins): add a way to inspect pytree kind more directly
    if tree == tree_flatten((core.unit,) * tree.num_leaves)[1]:
      lines1 = ["arg {} has shape {} and axis {} is to be mapped"
                .format(i, x.shape, d) for i, (x, d) in enumerate(zip(vals, dims))]
      sizes = collections.defaultdict(list)
      for i, (x, d) in enumerate(zip(vals, dims)):
        if d is not None:
          sizes[x.shape[d]].append(i)
      lines2 = ["{} {} {} {} to be mapped of size {}".format(
                  "args" if len(idxs) > 1 else "arg",
                  ", ".join(map(str, idxs)),
                  "have" if len(idxs) > 1 else "has",
                  "axes" if len(idxs) > 1 else "an axis",
                  size)
                for size, idxs in sizes.items()]
      raise ValueError(msg.format("\n".join(lines1 + ["so"] + lines2))) from e
    else:
      sizes = [x.shape[d] if d is not None else None for x, d in zip(vals, dims)]
      sizes = tree_unflatten(tree, sizes)
      raise ValueError(msg.format("the tree of axis sizes is:\n{}".format(sizes))) from e

def pmap(fun: Callable, axis_name: Optional[AxisName] = None, *, in_axes=0,
         static_broadcasted_argnums: Union[int, Iterable[int]] = (),
         devices=None, backend: Optional[str] = None,
         axis_size: Optional[int] = None,
         donate_argnums: Union[int, Iterable[int]] = ()) -> Callable:
  """Parallel map with support for collectives.

  The purpose of :py:func:`pmap` is to express single-program multiple-data (SPMD)
  programs. Applying :py:func:`pmap` to a function will compile the function with XLA
  (similarly to :py:func:`jit`), then execute it in parallel on XLA devices, such as
  multiple GPUs or multiple TPU cores. Semantically it is comparable to
  :py:func:`vmap` because both transformations map a function over array axes, but
  where :py:func:`vmap` vectorizes functions by pushing the mapped axis down into
  primitive operations, :py:func:`pmap` instead replicates the function and executes
  each replica on its own XLA device in parallel.

  Another key difference with :py:func:`vmap` is that while :py:func:`vmap` can only express
  pure maps, :py:func:`pmap` enables the use of parallel SPMD collective operations,
  like all-reduce sum.

  The mapped axis size must be less than or equal to the number of local XLA
  devices available, as returned by :py:func:`jax.local_device_count()` (unless
  ``devices`` is specified, see below). For nested :py:func:`pmap` calls, the product
  of the mapped axis sizes must be less than or equal to the number of XLA
  devices.

  **Multi-host platforms:** On multi-host platforms such as TPU pods, :py:func:`pmap`
  is designed to be used in SPMD Python programs, where every host is running
  the same Python code such that all hosts run the same pmapped function in the
  same order. Each host should still call the pmapped function with mapped axis
  size equal to the number of *local* devices (unless ``devices`` is specified,
  see below), and an array of the same leading axis size will be returned as
  usual. However, any collective operations in ``fun`` will be computed over
  *all* participating devices, including those on other hosts, via
  device-to-device communication.  Conceptually, this can be thought of as
  running a pmap over a single array sharded across hosts, where each host
  "sees" only its local shard of the input and output. The SPMD model requires
  that the same multi-host pmaps must be run in the same order on all devices,
  but they can be interspersed with arbitrary operations running on a single
  host.

  Args:
    fun: Function to be mapped over argument axes. Its arguments and return
      value should be arrays, scalars, or (nested) standard Python containers
      (tuple/list/dict) thereof. Positional arguments indicated by
      ``static_broadcasted_argnums`` can be anything at all, provided they are
      hashable and have an equality operation defined.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    in_axes: A nonnegative integer, None, or nested Python container thereof
      that specifies which axes in the input to map over (see :py:func:`vmap`).
      Currently, only 0 and None are supported axes for pmap.
    static_broadcasted_argnums: An int or collection of ints specifying which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded.
      Calling the pmaped function with different values for these constants will
      trigger recompilation. If the pmaped function is called with fewer
      positional arguments than indicated by ``static_argnums`` then an error is
      raised. Each of the static arguments will be broadcasted to all devices.
      Arguments that are not arrays or containers thereof must be marked as
      static. Defaults to ().
    devices: This is an experimental feature and the API is likely to change.
      Optional, a sequence of Devices to map over. (Available devices can be
      retrieved via jax.devices()). If specified, the size of the mapped axis
      must be equal to the number of local devices in the sequence. Nested
      :py:func:`pmap` s with ``devices`` specified in either the inner or outer :py:func:`pmap`
      are not yet supported.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.
    axis_size: Optional; the size of the mapped axis.
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perfom a computation, for
      example recycling one of your input buffers to store a result. You should
      not re-use buffers that you donate to a computation, JAX will raise an
      error if you try to.

  Returns:
    A parallelized version of ``fun`` with arguments that correspond to those of
    ``fun`` but with extra array axes at positions indicated by ``in_axes``
    and with output that has an additional leading array axis (with the same
    size).

  For example, assuming 8 XLA devices are available, :py:func:`pmap` can be used as a
  map along a leading array axis:

  >>> import jax.numpy as np
  >>>
  >>> out = pmap(lambda x: x ** 2)(np.arange(8))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [0, 1, 4, 9, 16, 25, 36, 49]

  When the leading dimension is smaller than the number of available devices JAX
  will simply run on a subset of devices:

  >>> x = np.arange(3 * 2 * 2.).reshape((3, 2, 2))
  >>> y = np.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
  >>> out = pmap(np.dot)(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [[[    4.     9.]
    [   12.    29.]]
   [[  244.   345.]
    [  348.   493.]]
   [[ 1412.  1737.]
    [ 1740.  2141.]]]

  If your leading dimension is larger than the number of available devices you
  will get an error:

  >>> pmap(lambda x: x ** 2)(np.arange(9))  # doctest: +SKIP
  ValueError: ... requires 9 replicas, but only 8 XLA devices are available

  As with :py:func:`vmap`, using ``None`` in ``in_axes`` indicates that an argument
  doesn't have an extra axis and should be broadcasted, rather than mapped,
  across the replicas:

  >>> x, y = np.arange(2.), 4.
  >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  ([4., 5.], [8., 8.])

  Note that :py:func:`pmap` always returns values mapped over their leading axis,
  equivalent to using ``out_axes=0`` in :py:func:`vmap`.

  In addition to expressing pure maps, :py:func:`pmap` can also be used to express
  parallel single-program multiple-data (SPMD) programs that communicate via
  collective operations. For example:

  >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
  >>> out = pmap(f, axis_name='i')(np.arange(4.))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [ 0.          0.16666667  0.33333334  0.5       ]
  >>> print(out.sum())  # doctest: +SKIP
  1.0

  In this example, ``axis_name`` is a string, but it can be any Python object
  with ``__hash__`` and ``__eq__`` defined.

  The argument ``axis_name`` to :py:func:`pmap` names the mapped axis so that
  collective operations, like :func:`jax.lax.psum`, can refer to it. Axis names
  are important particularly in the case of nested :py:func:`pmap` functions,
  where collectives can operate over distinct axes:

  >>> from functools import partial
  >>> import jax
  >>>
  >>> @partial(pmap, axis_name='rows')
  ... @partial(pmap, axis_name='cols')
  ... def normalize(x):
  ...   row_normed = x / jax.lax.psum(x, 'rows')
  ...   col_normed = x / jax.lax.psum(x, 'cols')
  ...   doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
  ...   return row_normed, col_normed, doubly_normed
  >>>
  >>> x = np.arange(8.).reshape((4, 2))
  >>> row_normed, col_normed, doubly_normed = normalize(x)  # doctest: +SKIP
  >>> print(row_normed.sum(0))  # doctest: +SKIP
  [ 1.  1.]
  >>> print(col_normed.sum(1))  # doctest: +SKIP
  [ 1.  1.  1.  1.]
  >>> print(doubly_normed.sum((0, 1)))  # doctest: +SKIP
  1.0

  On multi-host platforms, collective operations operate over all devices,
  including those those on other hosts. For example, assuming the following code
  runs on two hosts with 4 XLA devices each:

  >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
  >>> data = np.arange(4) if jax.host_id() == 0 else np.arange(4,8)
  >>> out = pmap(f, axis_name='i')(data)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [28 29 30 31] # on host 0
  [32 33 34 35] # on host 1

  Each host passes in a different length-4 array, corresponding to its 4 local
  devices, and the psum operates over all 8 values. Conceptually, the two
  length-4 arrays can be thought of as sharded length-8 array (in this example
  equivalent to np.arange(8)) that is mapped over, with the length-8 mapped axis
  given name 'i'. The pmap call on each host then returns the corresponding
  length-4 output shard.

  The ``devices`` argument can be used to specify exactly which devices are used
  to run the parallel computation. For example, again assuming a single host
  with 8 devices, the following code defines two parallel computations, one
  which runs on the first six devices and one on the remaining two:

  >>> from functools import partial
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
  ... def f1(x):
  ...   return x / jax.lax.psum(x, axis_name='i')
  >>>
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
  ... def f2(x):
  ...   return jax.lax.psum(x ** 2, axis_name='i')
  >>>
  >>> print(f1(np.arange(6.)))  # doctest: +SKIP
  [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
  >>> print(f2(np.array([2., 3.])))  # doctest: +SKIP
  [ 13.  13.]
  """
  # axis_size is an optional integer representing the global axis size.
  # The aggregate size (across all hosts) size of the mapped axis must match
  # the given value.

  _check_callable(fun)
  axis_name = _TempAxisName(fun) if axis_name is None else axis_name
  static_broadcasted_tuple = _ensure_tuple(static_broadcasted_argnums)
  donate_tuple = rebase_donate_argnums(_ensure_tuple(donate_argnums),
                                       static_broadcasted_tuple)

  if any(axis != 0 for axis in tree_leaves(in_axes)):
    raise ValueError(f"pmap in_axes leaves must be 0 or None, got {in_axes}")

  @wraps(fun)
  def f_pmapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    if static_broadcasted_tuple:
      if max(static_broadcasted_tuple) >= len(args):
        msg = ("pmapped function has static_broadcasted_argnums={} but was "
               "called with only {} positional argument{}. All static "
               "broadcasted arguments must be passed positionally.")
        raise ValueError(msg.format(static_broadcasted_tuple, len(args),
                                    "s" if len(args) > 1 else ""))
      dyn_argnums = [i for i in range(len(args))
                     if i not in static_broadcasted_tuple]
      f, dyn_args = argnums_partial(f, dyn_argnums, args)
      if isinstance(in_axes, tuple):
        dyn_in_axes = tuple(in_axes[i] for i in dyn_argnums)
      else:
        dyn_in_axes = in_axes
    else:
      dyn_args, dyn_in_axes = args, in_axes
    args, in_tree = tree_flatten((dyn_args, kwargs))
    donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
    in_axes_flat = flatten_axes("pmap in_axes", in_tree, (dyn_in_axes, 0))
    local_axis_size = _mapped_axis_size(in_tree, args, in_axes_flat, "pmap")
    for arg in args: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)
    out = pxla.xla_pmap(
        flat_fun, *args, backend=backend, axis_name=axis_name,
        axis_size=local_axis_size, global_axis_size=axis_size,
        devices=None if devices is None else tuple(devices),
        mapped_invars=tuple(axis is not None for axis in in_axes_flat),
        name=flat_fun.__name__, donated_invars=tuple(donated_invars))
    return tree_unflatten(out_tree(), out)

  return f_pmapped

class _TempAxisName:
  def __init__(self, obj):
    self.obj = id(obj)
    self.hash = hash(obj)
  def __repr__(self):
    return '<axis {}>'.format(hex(self.obj))
  def __hash__(self):
    return self.hash
  def __eq__(self, other):
    return type(other) is _TempAxisName and self.obj == other.obj


def soft_pmap(fun: Callable, axis_name: Optional[AxisName] = None, *,
              in_axes=0, backend: Optional[str] = None) -> Callable:
  warn("soft_pmap is an experimental feature and probably has bugs!")
  _check_callable(fun)
  axis_name = _TempAxisName(fun) if axis_name is None else axis_name

  if any(axis != 0 for axis in tree_leaves(in_axes)):
    raise ValueError(f"soft_pmap in_axes leaves must be 0 or None, got {in_axes}")

  @wraps(fun)
  def f_pmapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_axes_flat = flatten_axes("soft_pmap in_axes", in_tree, (in_axes, 0))
    mapped_invars = tuple(axis is not None for axis in in_axes_flat)
    axis_size = _mapped_axis_size(in_tree, args_flat, in_axes_flat, "soft_pmap")
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)

    chunk_size, leftover = divmod(axis_size, pxla.unmapped_device_count(backend))
    if chunk_size == 0 and leftover:
      return pmap(fun, axis_name, backend=backend)(*args)  # can map directly onto hardware
    elif leftover:
      msg = ("soft_pmap mapped axis size must be divisible by the number of "
             "XLA devices (or be less than or equal to that number), but got "
             "an axis size of {} with {} devices.")
      raise ValueError(msg.format(axis_size, pxla.unmapped_device_count()))
    num_chunks = axis_size // chunk_size

    reshaped_args = [_reshape_split(num_chunks, x) for x in args_flat]
    soft_mapped_fun = pxla.split_axis(flat_fun, axis_name, chunk_size)
    # TODO(tomhennigan): soft_pmap should support buffer donation.
    donated_invars = (False,) * len(reshaped_args)
    reshaped_outs = pxla.xla_pmap(soft_mapped_fun, *reshaped_args, backend=backend,
                                  axis_name=axis_name, axis_size=num_chunks,
                                  global_axis_size=None, devices=None,
                                  name=soft_mapped_fun.__name__,
                                  mapped_invars=mapped_invars,
                                  donated_invars=donated_invars)
    outs = [_reshape_merge(out) for out in reshaped_outs]
    return tree_unflatten(out_tree(), outs)
  return f_pmapped

def _reshape_split(num_chunks, x):
  aval = core.get_aval(x)
  if aval is core.abstract_unit:
    return x
  else:
    return x.reshape((num_chunks, x.shape[0] // num_chunks) + x.shape[1:])

def _reshape_merge(x):
  aval = core.get_aval(x)
  if aval is core.abstract_unit:
    return x
  else:
    return x.reshape((-1,) + x.shape[2:])


def _papply(fun):
  # This function is for testing purposes.
  axis_name = _TempAxisName(fun)

  def papply_fun(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(f, in_tree)
    axis_size = _mapped_axis_size(
        in_tree, args_flat, (0,) * len(args_flat), "papply")
    out_flat = parallel.papply(flat_fun, axis_name, args_flat, axis_size)
    return tree_unflatten(out_tree(), out_flat)

  return papply_fun, axis_name


def _parallelize(fun):
  axis_name = _TempAxisName(fun)

  def pfun(*args):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten(args)
    f, out_tree = flatten_fun_nokwargs(f, in_tree)
    axis_size = _mapped_axis_size(
        in_tree, args_flat, (0,) * len(args_flat), "parallelize")

    chunk_size, leftover = divmod(axis_size, pxla.unmapped_device_count())
    if chunk_size == 0 and leftover:
      return pmap(fun, axis_name)(*args)  # can map directly onto hardware
    elif leftover:
      raise ValueError
    num_chunks = axis_size // chunk_size

    reshaped_args = [_reshape_split(num_chunks, x) for x in args_flat]
    f, out_axes = parallel.papply_transform(f, axis_name, axis_size)
    f = pxla.split_axis(f, axis_name, chunk_size)
    outs = pxla.xla_pmap(f, *reshaped_args, backend=None, axis_name=axis_name,
                         axis_size=num_chunks, global_axis_size=None,
                         devices=None, name=f.__name__)
    outs = map(_reshape_merge, outs)
    outs = [batching.matchaxis(axis_size, 0, dst, x)
            for dst, x in zip(out_axes(), outs)]
    return tree_unflatten(out_tree(), outs)

  return pfun


def mask(fun: Callable, in_shapes, out_shape) -> Callable:
  _check_callable(fun)
  unique_ids = masking.UniqueIds()

  in_specs, in_shapes_tree = tree_flatten(in_shapes)
  in_specs = map(masking.parse_spec, in_specs)
  in_specs = map(partial(masking.remap_ids, unique_ids), in_specs)

  out_specs, out_spec_tree = tree_flatten(out_shape)
  out_specs = map(masking.parse_spec, out_specs)
  out_specs = map(partial(masking.remap_ids, unique_ids), out_specs)

  def wrapped_fun(args, logical_env):
    args_flat, in_tree = tree_flatten(args)
    if in_tree != in_shapes_tree:
      raise TypeError(f"Tree mismatch: Input {in_tree} and shape spec {in_shapes_tree}.")
    logical_env = {unique_ids[name] : val for name, val in logical_env.items()}
    in_shapes = map(masking.finalize_spec, in_specs, map(onp.shape, args_flat))
    padded_env = masking.bind_shapes(in_shapes, [x.shape for x in args_flat])
    f = lu.wrap_init(fun)
    flat_fun, out_tree_thunk = flatten_fun_nokwargs(f, in_tree)
    outs, out_shapes = masking.mask_fun(
      flat_fun, logical_env, padded_env, args_flat, in_shapes)
    out_tree = out_tree_thunk()

    masking.check_shapes(out_specs, out_spec_tree, list(out_shapes), out_tree)
    def padded_spec(shape_spec):
      return tuple(dim if dim is masking._monomorphic_dim else
                   masking.eval_poly(dim, padded_env) for dim in shape_spec)
    masking.check_shapes(map(padded_spec, out_specs), out_spec_tree,
                         map(onp.shape, outs), out_tree, "Padded output")
    return tree_unflatten(out_tree, outs)
  return wrapped_fun

@curry
def shapecheck(in_shapes, out_shape, fun: Callable):
  _check_callable(fun)
  in_shapes, in_tree = tree_flatten(in_shapes)
  in_shapes = map(masking.parse_spec, in_shapes)
  out_specs, out_spec_tree = tree_flatten(out_shape)
  out_specs = map(masking.parse_spec, out_specs)
  flat_fun, out_tree_thunk = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  avals = map(partial(ShapedArray, dtype=onp.float32), in_shapes)
  out_shapes = [o.shape for o in pe.abstract_eval_fun(flat_fun.call_wrapped, *avals)]
  masking.check_shapes(map(tuple, out_specs), out_spec_tree,
                       map(tuple, out_shapes), out_tree_thunk())
  return fun

def jvp(fun: Callable, primals, tangents) -> Tuple[Any, Any]:
  """Computes a (forward-mode) Jacobian-vector product of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be either a tuple or a list of arguments,
      and its length should  equal to the number of positional parameters of
      ``fun``.
    tangents: The tangent vector for which the Jacobian-vector product should be
      evaluated. Should be either a tuple or a list of tangents, with the same
      tree structure and array shapes as ``primals``.

  Returns:
    A ``(primals_out, tangents_out)`` pair, where ``primals_out`` is
    ``fun(*primals)``, and ``tangents_out`` is the Jacobian-vector product of
    ``function`` evaluated at ``primals`` with ``tangents``. The
    ``tangents_out`` value has the same Python tree structure and shapes as
    ``primals_out``.

  For example:

  >>> import jax
  >>>
  >>> y, v = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  >>> print(y)
  0.09983342
  >>> print(v)
  0.19900084
  """
  _check_callable(fun)
  return _jvp(lu.wrap_init(fun), primals, tangents)

def _jvp(fun: lu.WrappedFun, primals, tangents):
  """Variant of jvp() that takes an lu.WrappedFun."""
  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    msg = ("primal and tangent arguments to jax.jvp must be tuples or lists; "
           "found {} and {}.")
    raise TypeError(msg.format(type(primals).__name__, type(tangents).__name__))

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    msg = ("primal and tangent arguments to jax.jvp must have the same tree "
           "structure; primals have tree structure {} whereas tangents have "
           "tree structure {}")
    raise TypeError(msg.format(tree_def, tree_def_2))
  for p, t in safe_zip(ps_flat, ts_flat):
    if _dtype(p) != _dtype(t):
      msg = ("primal and tangent arguments to jax.jvp must have equal types; "
             "type mismatch primal {} vs tangent {}")
      raise TypeError(msg.format(_dtype(p), _dtype(t)))
  flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
  out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
  return (tree_unflatten(out_tree(), out_primals),
          tree_unflatten(out_tree(), out_tangents))

def linearize(fun: Callable, *primals) -> Tuple[Any, Callable]:
  """Produces a linear approximation to ``fun`` using :py:func:`jvp` and partial eval.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be a tuple of arrays, scalar, or standard Python
      container thereof. The length of the tuple is equal to the number of
      positional parameters of ``fun``.

  Returns:
    A pair where the first element is the value of ``f(*primals)`` and the
    second element is a function that evaluates the (forward-mode)
    Jacobian-vector product of ``fun`` evaluated at ``primals`` without re-doing
    the linearization work.

  In terms of values computed, :py:func:`linearize` behaves much like a curried
  :py:func:`jvp`, where these two code blocks compute the same values::

    y, out_tangent = jax.jvp(f, (x,), (in_tangent,))

    y, f_jvp = jax.linearize(f, x)
    out_tangent = f_jvp(in_tangent)

  However, the difference is that :py:func:`linearize` uses partial evaluation
  so that the function ``f`` is not re-linearized on calls to ``f_jvp``. In
  general that means the memory usage scales with the size of the computation,
  much like in reverse-mode. (Indeed, :py:func:`linearize` has a similar
  signature to :py:func:`vjp`!)

  This function is mainly useful if you want to apply ``f_jvp`` multiple times,
  i.e. to evaluate a pushforward for many different input tangent vectors at the
  same linearization point. Moreover if all the input tangent vectors are known
  at once, it can be more efficient to vectorize using :py:func:`vmap`, as in::

    pushfwd = partial(jvp, f, (x,))
    y, out_tangents = vmap(pushfwd, out_axes=(None, 0))((in_tangents,))

  By using :py:func:`vmap` and :py:func:`jvp` together like this we avoid the stored-linearization
  memory cost that scales with the depth of the computation, which is incurred
  by both :py:func:`linearize` and :py:func:`vjp`.

  Here's a more complete example of using :py:func:`linearize`:

  >>> import jax
  >>> import jax.numpy as np
  >>>
  >>> def f(x): return 3. * np.sin(x) + np.cos(x / 2.)
  ...
  >>> jax.jvp(f, (2.,), (3.,))
  (DeviceArray(3.26819, dtype=float32), DeviceArray(-5.00753, dtype=float32))
  >>> y, f_jvp = jax.linearize(f, 2.)
  >>> print(y)
  3.2681944
  >>> print(f_jvp(3.))
  -5.007528
  >>> print(f_jvp(4.))
  -6.676704
  """
  _check_callable(fun)
  f = lu.wrap_init(fun)
  primals_flat, in_tree = tree_flatten((primals, {}))
  jaxtree_fun, out_tree = flatten_fun(f, in_tree)
  out_primals, out_pvals, jaxpr, consts = ad.linearize(jaxtree_fun, *primals_flat)
  out_tree = out_tree()
  out_primal_py = tree_unflatten(out_tree, out_primals)
  primal_avals = list(map(core.get_aval, primals_flat))
  lifted_jvp = partial(_lift_linearized, jaxpr, primal_avals, consts,
                       (in_tree, out_tree), out_pvals)
  return out_primal_py, lifted_jvp

def _lift_linearized(jaxpr, primal_avals, consts, io_tree, out_pvals, *py_args):
  def fun(*tangents):
    tangent_avals = list(map(core.get_aval, tangents))
    for primal_aval, tangent_aval in zip(primal_avals, tangent_avals):
      try:
        core.lattice_join(primal_aval, tangent_aval)
      except TypeError as e:
        msg = ("linearized function called on tangent values inconsistent with "
               "the original primal values.")
        raise ValueError(msg) from e
    tangents_out = eval_jaxpr(jaxpr, consts, *tangents)
    return tuple(map(lambda out_pv, tan_out: out_pv.merge_with_known(tan_out),
                     out_pvals, tangents_out))

  return apply_flat_fun(fun, io_tree, *py_args)

def _check_inexact_input_vjp(x):
  aval = core.get_aval(x)
  if not dtypes.issubdtype(aval.dtype, onp.inexact):
    msg = ("Primal inputs to reverse-mode differentiation must be of float "
           "or complex type, got type {}")
    raise TypeError(msg.format(aval.dtype.name))

def _vjp_pullback_wrapper(fun, cotangent_dtypes, io_tree, py_args):
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    msg = ("Tree structure of cotangent input {}, does not match structure of "
           "primal output {}")
    raise TypeError(msg.format(in_tree, in_tree_expected))
  for a, dtype in safe_zip(args, cotangent_dtypes):
    if _dtype(a) != dtype:
      msg = ("Type of cotangent input to vjp pullback function ({}) does not "
             "match type of corresponding primal output ({})")
      raise TypeError(msg.format(_dtype(a), dtype))
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)


def vjp(fun: Callable, *primals, **kwargs
        ) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
  """Compute a (reverse-mode) vector-Jacobian product of ``fun``.

  :py:func:`grad` is implemented as a special case of :py:func:`vjp`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: A sequence of primal values at which the Jacobian of ``fun``
      should be evaluated. The length of ``primals`` should be equal to the
      number of positional parameters to ``fun``. Each primal value should be a
      tuple of arrays, scalar, or standard Python containers thereof.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, vjpfun)`` pair, where
    ``primals_out`` is ``fun(*primals)``.
    ``vjpfun`` is a function from a cotangent vector with the same shape as
    ``primals_out`` to a tuple of cotangent vectors with the same shape as
    ``primals``, representing the vector-Jacobian product of ``fun`` evaluated at
    ``primals``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, vjpfun, aux)`` tuple where ``aux`` is the auxiliary data
    returned by ``fun``.

  >>> import jax
  >>>
  >>> def f(x, y):
  ...   return jax.numpy.sin(x), jax.numpy.cos(y)
  ...
  >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
  >>> xbar, ybar = f_vjp((-0.7, 0.3))
  >>> print(xbar)
  -0.61430776
  >>> print(ybar)
  -0.2524413
  """
  _check_callable(fun)
  return _vjp(lu.wrap_init(fun), *primals, **kwargs)

def _vjp(fun: lu.WrappedFun, *primals, **kwargs):
  """Variant of vjp() that takes an lu.WrappedFun."""
  has_aux = kwargs.pop('has_aux', False)
  assert not kwargs
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: _check_arg(arg)
  tree_map(_check_inexact_input_vjp, primals)
  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    out_primal, out_vjp = ad.vjp(flat_fun, primals_flat)
    out_tree = out_tree()
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, in_tree)
    out_primal, out_vjp, aux = ad.vjp(flat_fun, primals_flat, has_aux=True)
    out_tree, aux_tree = out_aux_trees()
  out_primal_py = tree_unflatten(out_tree, out_primal)
  vjp_py = partial(_vjp_pullback_wrapper, out_vjp,
                   [_dtype(x) for x in out_primal], (out_tree, in_tree))
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)


def make_jaxpr(fun: Callable,
               static_argnums: Union[int, Iterable[int]] = ()
               ) -> Callable[..., core.TypedJaxpr]:
  """Creates a function that produces its jaxpr given example args.

  Args:
    fun: The function whose ``jaxpr`` is to be computed. Its positional
      arguments and return value should be arrays, scalars, or standard Python
      containers (tuple/list/dict) thereof.
    static_argnums: See the :py:func:`jax.jit` docstring.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
      a ``TypedJaxpr`` representation of ``fun`` on those arguments.

  A ``jaxpr`` is JAX's intermediate representation for program traces. The
  ``jaxpr`` language is based on the simply-typed first-order lambda calculus
  with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
  ``jaxpr``, which we can inspect to understand what JAX is doing internally.
  The ``jaxpr`` returned is a trace of ``fun`` abstracted to
  :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

  We do not describe the semantics of the ``jaxpr`` language in detail here, but
  instead give a few examples.

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> print(f(3.0))
  -0.83602
  >>> jax.make_jaxpr(f)(3.0)
  { lambda  ; a.
    let b = cos a
        c = sin b
    in (c,) }
  >>> jax.make_jaxpr(jax.grad(f))(3.0)
  { lambda  ; a.
    let b = cos a
        c = cos b
        d = mul 1.0 c
        e = neg d
        f = sin a
        g = mul e f
    in (g,) }
  """
  _check_callable(fun)
  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  def pv_like(x):
    aval = xla.abstractify(x)
    return pe.PartialVal.unknown(aval)

  @wraps(fun)
  def jaxpr_maker(*args, **kwargs):
    wrapped = lu.wrap_init(fun)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      wrapped, _ = argnums_partial(wrapped, dyn_argnums, args)
    jax_args, in_tree = tree_flatten((args, kwargs))
    jaxtree_fun, out_tree = flatten_fun(wrapped, in_tree)
    in_pvals = map(pv_like, jax_args)
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr(
        jaxtree_fun, in_pvals, instantiate=True, stage_out=True)
    out_avals = map(raise_to_shaped, unzip2(out_pvals)[0])
    in_avals = tuple(raise_to_shaped(in_aval) for in_aval, _ in in_pvals)
    typed_jaxpr = core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)
    return typed_jaxpr

  jaxpr_maker.__name__ = "make_jaxpr({})".format(jaxpr_maker.__name__)
  return jaxpr_maker


def device_put(x, device: Optional[xc.Device] = None):
  """Transfers ``x`` to ``device``.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    device: The (optional) :py:class:`Device` to which ``x`` should be
      transferred. If given, then the result is committed to the device.

  If the ``device`` parameter is ``None``, then this operation behaves like the
  identity function if the operand is on any device already, otherwise it
  transfers the data to the default device, uncommitted.

  For more details on data placement see the
  :ref:`FAQ on data placement <faq-data-placement>`.

  Returns:
    A copy of ``x`` that resides on ``device``.
  """
  return tree_map(lambda y: xla.device_put_p.bind(y, device=device), x)


# TODO(mattjj): consider revising
def _device_get(x):
  if isinstance(x, core.Tracer):
    return x
  return x.copy()

def device_get(x):
  for y in tree_leaves(x):
    try:
      y.copy_to_host_async()
    except AttributeError:
      pass
  return tree_map(_device_get, x)


def _check_arg(arg):
  if not (isinstance(arg, core.Tracer) or _valid_jaxtype(arg)):
    raise TypeError("Argument '{}' of type {} is not a valid JAX type"
                    .format(arg, type(arg)))

# TODO(necula): this duplicates code in core.valid_jaxtype
def _valid_jaxtype(arg):
  try:
    xla.abstractify(arg)  # faster than core.get_aval
  except TypeError:
    return False
  else:
    return True


class ShapeDtypeStruct(object):
  __slots__ = ["shape", "dtype"]
  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = onp.dtype(dtype)

  size = property(lambda self: onp.prod(self.shape))
  ndim = property(lambda self: len(self.shape))

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as e:
      raise TypeError("len() of unsized object") from e # same as numpy error

  def __repr__(self):
    return "{}(shape={}, dtype={})".format(
        type(self).__name__, self.shape, self.dtype.name)

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, ShapeDtypeStruct):
      return False
    else:
      return (other.shape, other.dtype) == (self.shape, self.dtype)

  def __hash__(self):
    return hash((self.shape, self.dtype))

def eval_shape(fun: Callable, *args, **kwargs):
  """Compute the shape/dtype of ``fun`` without any FLOPs.

  This utility function is useful for performing shape inference. Its
  input/output behavior is defined by::

    def eval_shape(fun, *args, **kwargs):
      out = fun(*args, **kwargs)
      return jax.tree_util.tree_map(shape_dtype_struct, out)

    def shape_dtype_struct(x):
      return ShapeDtypeStruct(x.shape, x.dtype)

    class ShapeDtypeStruct(object):
      __slots__ = ["shape", "dtype"]
      def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

  In particular, the output is a pytree of objects that have ``shape`` and
  ``dtype`` attributes, but nothing else about them is guaranteed by the API.

  But instead of applying ``fun`` directly, which might be expensive, it uses
  JAX's abstract interpretation machinery to evaluate the shapes without doing
  any FLOPs.

  Using :py:func:`eval_shape` can also catch shape errors, and will raise same
  shape errors as evaluating ``fun(*args, **kwargs)``.

  Args:
    fun: The function whose output shape should be evaluated.
    *args: a positional argument tuple of arrays, scalars, or (nested) standard
      Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
      those types. Since only the ``shape`` and ``dtype`` attributes are
      accessed, only values that duck-type arrays are required, rather than real
      ndarrays. The duck-typed objects cannot be namedtuples because those are
      treated as standard Python containers. See the example below.
    **kwargs: a keyword argument dict of arrays, scalars, or (nested) standard
      Python containers (pytrees) of those types. As in ``args``, array values
      need only be duck-typed to have ``shape`` and ``dtype`` attributes.

  For example:

  >>> import jax
  >>> import jax.numpy as np
  >>>
  >>> f = lambda A, x: np.tanh(np.dot(A, x))
  >>> class MyArgArray(object):
  ...   def __init__(self, shape, dtype):
  ...     self.shape = shape
  ...     self.dtype = dtype
  ...
  >>> A = MyArgArray((2000, 3000), np.float32)
  >>> x = MyArgArray((3000, 1000), np.float32)
  >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
  >>> print(out.shape)
  (2000, 1000)
  >>> print(out.dtype)
  float32
  """
  def abstractify(x):
    return ShapedArray(onp.shape(x), dtypes.result_type(x))
  args_flat, in_tree = tree_flatten((args, kwargs))
  wrapped_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  out = pe.abstract_eval_fun(wrapped_fun.call_wrapped,
                             *map(abstractify, args_flat))
  out = [ShapeDtypeStruct(x.shape, x.dtype) for x in out]
  return tree_unflatten(out_tree(), out)


def checkpoint(fun: Callable, concrete: bool = False) -> Callable:
  """Make ``fun`` recompute internal linearization points when differentiated.

  The :func:`jax.checkpoint` decorator, aliased to ``jax.remat``, provides a
  way to trade off computation time and memory cost in the context of automatic
  differentiation, especially with reverse-mode autodiff like :func:`jax.grad`
  and :func:`jax.vjp` but also with :func:`jax.linearize`.

  When differentiating a function in reverse-mode, by default all the
  linearization points (e.g. inputs to elementwise nonlinear primitive
  operations) are stored when evaluating the forward pass so that they can be
  reused on the backward pass. This evaluation strategy can lead to a high
  memory cost, or even to poor performance on hardware acceleartors where memory
  access is much more expensive than FLOPs.

  An alternative evaluation strategy is for some of the linearization points to
  be recomputed (i.e. rematerialized) rather than stored. This approach can
  reduce memory usage at the cost of increased computation.

  This function decorator produces a new version of ``fun`` which follows
  the rematerialization strategy rather than the default store-everything
  strategy. That is, it returns a new version of ``fun`` which, when
  differentiated, doesn't store any of its intermediate linearization points.
  Instead, these linearization points are recomputed from the function's saved
  inputs.

  See the examples below.

  Args:
    fun: Function for which the autodiff evaluation strategy is to be changed
      from the default of storing all intermediate linearization points to
      recomputing them. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
    concrete: Optional, boolean indicating whether ``fun`` may involve
      value-dependent Python control flow (default False). Support for such
      control flow is optional, and disabled by default, because in some
      edge-case compositions with :func:`jax.jit` it can lead to some extra
      computation.

  Returns:
    A function (callable) with the same input/output behavior as ``fun`` but
    which, when differentiated using e.g. :func:`jax.grad`, :func:`jax.vjp`, or
    :func:`jax.linearize`, recomputes rather than stores intermediate
    linearization points, thus potentially saving memory at the cost of extra
    computation.

  Here is a simple example:

  >>> import jax
  >>> import jax.numpy as jnp

  >>> @jax.checkpoint
  ... def g(x):
  ...   y = jnp.sin(x)
  ...   z = jnp.sin(y)
  ...   return z
  ...
  >>> jax.grad(g)(2.0)
  DeviceArray(-0.25563914, dtype=float32)

  Here, the same value is produced whether or not the :func:`jax.checkpoint`
  decorator is present. But when using :func:`jax.checkpoint`, the value
  ``jnp.sin(2.0)`` is computed twice: once on the forward pass, and once on the
  backward pass. The values ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))`` are
  also computed twice. Without using the decorator, both ``jnp.cos(2.0)`` and
  ``jnp.cos(jnp.sin(2.0))`` would be stored and reused.

  The :func:`jax.checkpoint` decorator can be applied recursively to express
  sophisticated autodiff rematerialization strategies. For example:

  >>> def recursive_checkpoint(funs):
  ...   if len(funs) == 1:
  ...     return funs[0]
  ...   elif len(funs) == 2:
  ...     f1, f2 = funs
  ...     return lambda x: f1(f2(x))
  ...   else:
  ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
  ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
  ...     return lambda x: f1(jax.checkpoint(f2)(x))
  ...
  """
  @wraps(fun)
  def fun_remat(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    out_flat = pe.remat_call(flat_fun, *args_flat, name=flat_fun.__name__,
                             concrete=concrete)
    return tree_unflatten(out_tree(), out_flat)
  return fun_remat
remat = checkpoint


# TODO(mattjj): delete everything below here (deprecated custom_transforms)

class CustomTransformsFunction(object):
  def __init__(self, fun, prim):
    self.fun = fun
    self.prim = prim
    wraps(fun)(self)

  def __repr__(self):
    return '<jax.custom_transforms function {fun}>'.format(fun=self.__name__)

  def __call__(self, *args):
    # TODO(mattjj): instead of tracing to a jaxpr, use process_call
    args_flat, in_tree = tree_flatten(args)
    flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    in_pvals = [pe.PartialVal.unknown(raise_to_shaped(core.get_aval(x)))
                for x in args_flat]
    with core.initial_style_staging():
      jaxpr, _, consts = pe.trace_to_jaxpr(flat_fun, in_pvals, instantiate=True)
    outs = self.prim.bind(*it.chain(consts, args_flat), jaxpr=jaxpr,
                          in_tree=in_tree, out_tree=out_tree(),
                          num_consts=len(consts))
    return tree_unflatten(out_tree(), outs)

def custom_transforms(fun):
  """Deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp`."""

  name = getattr(fun, '__name__', '<unnamed custom_transforms primitive>')
  fun_p = core.Primitive(name)
  fun_p.multiple_results = True

  def fun_impl(*args, **params):
    consts, args = split_list(args, [params['num_consts']])
    return core.eval_jaxpr(params['jaxpr'], consts, *args)
  fun_p.def_impl(fun_impl)

  def fun_jvp(primals, tangents, **params):
    return ad.jvp(lu.wrap_init(fun_impl, params)).call_wrapped(primals, tangents)
  ad.primitive_jvps[fun_p] = fun_jvp

  def fun_batch(args, dims, **params):
    batched, out_dims = batching.batch_fun2(lu.wrap_init(fun_impl, params), dims)
    return batched.call_wrapped(*args), out_dims()
  batching.primitive_batchers[fun_p] = fun_batch

  def fun_abstract_eval(*avals, **params):
    return pe.abstract_eval_fun(fun_impl, *avals, **params)
  fun_p.def_abstract_eval(fun_abstract_eval)

  def fun_translation(c, *xla_args, **params):
    return xla.lower_fun(fun_impl, multiple_results=True)(c, *xla_args, **params)
  xla.translations[fun_p] = fun_translation

  return CustomTransformsFunction(fun, fun_p)

def _check_custom_transforms_type(name, fun):
  if type(fun) is not CustomTransformsFunction:
    msg = ("{} requires a custom_transforms function as its first argument, "
          "but got type {}.")
    raise TypeError(msg.format(name, type(fun)))

def defjvp_all(fun, custom_jvp):
  """This API is deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp` instead."""

  _check_custom_transforms_type("defjvp_all", fun)
  def custom_transforms_jvp(primals, tangents, **params):
    num_consts, in_tree = params['num_consts'], params['in_tree']
    _, args_flat = split_list(primals, [num_consts])
    consts_dot, args_dot_flat = split_list(tangents, [num_consts])
    if not all(type(t) is ad_util.Zero for t in consts_dot):
      msg = ("Detected differentiation with respect to closed-over values with "
             "custom JVP rule, which isn't supported.")
      raise ValueError(msg)
    args_dot_flat = map(ad.instantiate_zeros, args_dot_flat)
    args = tree_unflatten(in_tree, args_flat)
    args_dot = tree_unflatten(in_tree, args_dot_flat)
    out, out_dot = custom_jvp(args, args_dot)
    out_flat, out_tree = tree_flatten(out)
    out_dot_flat, out_tree2 = tree_flatten(out_dot)
    if out_tree != out_tree2:
      msg = ("Custom JVP rule returned different tree structures for primals "
             "and tangents, but they must be equal: {} and {}.")
      raise TypeError(msg.format(out_tree, out_tree2))
    return out_flat, out_dot_flat
  ad.primitive_jvps[fun.prim] = custom_transforms_jvp

def defjvp(fun, *jvprules):
  """This API is deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp` instead."""

  _check_custom_transforms_type("defjvp", fun)
  def custom_jvp(primals, tangents):
    ans = fun(*primals)
    tangents_out = [rule(t, ans, *primals) for rule, t in zip(jvprules, tangents)
                    if rule is not None and type(t) is not ad_util.Zero]
    return ans, functools.reduce(ad.add_tangents, tangents_out, ad_util.Zero.from_value(ans))
  defjvp_all(fun, custom_jvp)

def defvjp_all(fun, custom_vjp):
  """This API is deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp` instead."""

  _check_custom_transforms_type("defvjp_all", fun)
  def custom_transforms_vjp(*consts_and_args, **params):
    num_consts, in_tree = params['num_consts'], params['in_tree']
    consts, args_flat = split_list(consts_and_args, [num_consts])
    args = tree_unflatten(params['in_tree'], args_flat)
    out, vjp = custom_vjp(*args)
    out_flat, out_tree = tree_flatten(out)
    if out_tree != params['out_tree']:
      msg = (
        "First output of `custom_vjp`: {} doesn't match the structure of "
        "the output of `fun`: {}\n"
        "{}\n"
        "vs\n"
        "{}\n".format(custom_vjp, fun, out_tree, params['out_tree'])
      )
      raise TypeError(msg)
    def vjp_flat(*cts_flat):
      cts = tree_unflatten(out_tree, cts_flat)
      args_cts_flat, in_tree2 = tree_flatten(vjp(cts))
      if in_tree != in_tree2:
        msg = (
          "Output of the `vjp`: {} doesn't match the structure of args of "
          "`fun`: {}\n"
          "{}\n"
          "vs\n"
          "{}\n".format(vjp, fun, in_tree2, in_tree)
        )
        raise TypeError(msg)
      return [core.unit] * num_consts + list(args_cts_flat)
    return out_flat, vjp_flat
  ad.defvjp_all(fun.prim, custom_transforms_vjp)

def defvjp(fun, *vjprules):
  """This API is deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp` instead."""

  _check_custom_transforms_type("defvjp", fun)
  def custom_vjp(*primals):
    ans = fun(*primals)
    # TODO(mattjj): avoid instantiating zeros?
    def vjpfun(ct):
      return tuple(vjp(ct, ans, *primals) if vjp else ad_util.zeros_like_jaxval(x)
                   for x, vjp in zip(primals, vjprules))
    return ans, vjpfun
  defvjp_all(fun, custom_vjp)

def custom_gradient(fun):
  """This API is deprecated. See :py:func:`jax.custom_jvp` and :py:func:`jax.custom_vjp` instead."""

  def primal_fun(*args, **kwargs):
    ans, _ = fun(*args, **kwargs)
    return ans
  primal_fun = custom_transforms(primal_fun)
  defvjp_all(primal_fun, fun)
  return primal_fun

def _ensure_tuple(x: Union[int, Iterable[int]]) -> Tuple[int, ...]:
  return (x,) if isinstance(x, int) else tuple(x)

def invertible(fun: Callable, concrete: bool = False) -> Callable:
  """Asserts that the decorated function is invertible.

  Applying reverse-mode AD to a decorated function will use a more memory efficient
  procedure than usual, which will reconstruct the necessary intermediate values
  by inverting the function. Note that this might degrade the numerical accuracy of
  obtained gradients if the inverse is unstable.

  Args:
    fun: The function assumed to be invertible.
    concrete: See the documentation of checkpoint.
  """
  @wraps(fun)
  def fun_invertible(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    out_flat = iad.invertible_call(flat_fun, *args_flat, name=flat_fun.__name__,
                                   concrete=concrete)
    return tree_unflatten(out_tree(), out_flat)
  return fun_invertible
