# Copyright 2019 Google LLC
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

"""Loops is an **experimental** module for syntactic sugar for loops and control-flow.

The current implementation should convert loops correctly to JAX internal
representation, and most transformations should work (see below), but we have
not yet fine-tuned the performance of the resulting XLA compilation!

By default, loops and control-flow in JAX are executed and inlined during tracing.
For example, in the following code the `for` loop is unrolled during JAX tracing::

  arr = onp.zeros(5)
  for i in range(arr.shape[0]):
    arr[i] += 2.
    if i % 2 == 0:
      arr[i] += 1.

In order to capture the structured control-flow one has to use the higher-order
JAX operations, which require you to express the body of the loops and
conditionals as functions, and the array updates using a functional style that
returns an updated array, e.g.::

  arr = onp.zeros(5)
  def loop_body(i, acc_arr):
    arr1 = ops.index_update(acc_arr, i, acc_arr[i] + 2.)
    return lax.cond(i % 2 == 0,
                    arr1,
                    lambda arr1: ops.index_update(arr1, i, arr1[i] + 1),
                    arr1,
                    lambda arr1: arr1)
  arr = lax.fori_loop(0, arr.shape[0], loop_body, arr)

The default notation quickly gets unreadable with deeper nested loops.
With the utilities in this module you can write loops and conditionals that
look closer to plain Python, as long as you keep the loop-carried state in a
special `loops.scope` object and use `for` loops over special
`scope.range` iterators::

  from jax.experimental import loops
  with loops.Scope() as s:
    s.arr = np.zeros(5)  # Create the mutable state of the loop as `scope` fields.
    for i in s.range(s.arr.shape[0]):
      s.arr = ops.index_update(s.arr, i, s.arr[i] + 2.)
      for _ in s.cond_range(i % 2 == 0):  # Conditionals as loops with 0 or 1 iterations
        s.arr = ops.index_update(s.arr, i, s.arr[i] + 1.)

Loops constructed with `range` must have literal constant bounds. If you need
loops with dynamic bounds, you can use the more general `while_range` iterator.
However, in that case that `grad` transformation is not supported::

    s.idx = start
    for _ in s.while_range(lambda: s.idx < end):
      s.idx += 1

Notes:
  * Loops and conditionals to be functionalized can appear only inside scopes
    constructed with `loops.Scope` and they must use one of the `Scope.range`
    iterators. All other loops are unrolled during tracing, as usual in JAX.
  * Only scope data (stored in fields of the scope object) is functionalized.
    All other state, e.g., in other Python variables, will not be considered as
    being part of the loop output. All references to the mutable state should be
    through the scope: `s.arr`.
  * Conceptually, this model is still "functional" in the sense that a loop over
    a `Scope.range` behaves as a function whose input and output is the scope data.
  * Scopes should be passed down to callees that need to use loop
    functionalization, or they may be nested.
  * The programming model is that the loop body over a `scope.range` is traced
    only once, using abstract shape values, similar to how JAX traces function
    bodies.

Restrictions:
  * The tracing of the loop body should not exit prematurely with `return`,
    `exception`, `break`. This would be detected and reported as errors when we
     encounter unnested scopes.
  * The loop index variable should not be used after the loop. Similarly, one
    should not use outside the loop data computed in the loop body, except data
    stored in fields of the scope object.
  * No new mutable state can be created inside a loop to be functionalized.
    All mutable state must be created outside all loops and conditionals.
  * For a `while` loop, the conditional function is not allowed to modify the
    scope state. This is a checked error. Also, for `while` loops the `grad`
    transformation does not work. An alternative that allows `grad` is a bounded
    loop (`range`).

Transformations:
  * All transformations are supported, except `grad` is not supported for
    `Scope.while_range` loops.
  * `vmap` is very useful for such loops because it pushes more work into the
    inner-loops, which should help performance for accelerators.

For usage example, see tests/loops_test.py.
"""


import copy
from functools import partial
import itertools
import numpy as onp
import traceback
from typing import Any, List, cast

from jax import abstract_arrays
from jax import lax, core
from jax.lax import lax_control_flow
from jax import tree_util
from jax import numpy as jnp
from jax.interpreters import partial_eval as pe
from jax.util import unzip2, safe_map


class Scope(object):
  """A scope context manager to keep the state of loop bodies for functionalization.

  Usage::

    with Scope() as s:
      s.data = 0.
      for i in s.range(5):
        s.data += 1.
      return s.data

  """

  def __init__(self):
    self._mutable_state = {}  # state to be functionalized, indexed by name.
    self._active_ranges = []  # stack of active ranges, last one is the innermost.
    self._count_subtraces = 0  # How many net started subtraces, for error recovery

  def range(self, first, second=None, third=None):
    """Creates an iterator for bounded iterations to be functionalized.

    The body is converted to a `lax.scan`, for which all JAX transformations work.
    The `first`, `second`, and `third` arguments must be integer literals.

    Usage::

      range(5)  # start=0, end=5, step=1
      range(1, 5)  # start=1, end=5, step=1
      range(1, 5, 2)  # start=1, end=5, step=2

      s.out = 1.
      for i in scope.range(5):
        s.out += 1.
    """
    if third is not None:
      start = int(first)
      stop = int(second)
      step = int(third)
    else:
      step = 1
      if second is not None:
        start = int(first)
        stop = int(second)
      else:
        start = 0
        stop = int(first)
    return _BodyTracer(self, _BoundedLoopBuilder(start, stop, step))

  def cond_range(self, pred):
    """Creates a conditional iterator with 0 or 1 iterations based on the boolean.

    The body is converted to a `lax.cond`. All JAX transformations work.

    Usage::

      for _ in scope.cond_range(s.field < 0.):
        s.field = - s.field
    """
    # TODO: share these checks with lax_control_flow.cond
    if len(onp.shape(pred)) != 0:
      raise TypeError(
        "Pred must be a scalar, got {} of shape {}.".format(pred, onp.shape(pred)))

    try:
      pred_dtype = onp.result_type(pred)
    except TypeError as err:
      msg = ("Pred type must be either boolean or number, got {}.")
      raise TypeError(msg.format(pred)) from err

    if pred_dtype.kind != 'b':
      if pred_dtype.kind in 'iuf':
        pred = pred != 0
      else:
        msg = ("Pred type must be either boolean or number, got {}.")
        raise TypeError(msg.format(pred_dtype))

    return _BodyTracer(self, _CondBuilder(pred))

  def while_range(self, cond_func):
    """Creates an iterator that continues as long as `cond_func` returns true.

    The body is converted to a `lax.while_loop`.
    The `grad` transformation does not work.

    Usage::

      for _ in scope.while_range(lambda: s.loss > 1.e-5):
        s.loss = loss(...)

    Args:
      cond_func: a lambda with no arguments, the condition for the "while".
    """
    return _BodyTracer(self, _WhileBuilder(cond_func))

  def _push_range(self, range_):
    for ar in self._active_ranges:
      if ar is range_:
        raise ValueError("Range is reused nested inside itself.")
    self._active_ranges.append(range_)

  def _pop_range(self, range_):
    if not (range_ is self._active_ranges[-1]):
      self._error_premature_exit_range()
    self._active_ranges.pop()

  def _error_premature_exit_range(self):
    """Raises error about premature exit from a range"""
    msg = "Some ranges have exited prematurely. The innermost such range is at\n{}"
    raise ValueError(msg.format(self._active_ranges[-1].location()))

  def __getattr__(self, key):
    """Accessor for scope data.

    Called only if the attribute is not found, which will happen when we read
    scope data that has been stored in self._mutable_state.
    """
    mt_val = self._mutable_state.get(key)
    if mt_val is None:
      raise AttributeError(
        "Reading uninitialized data '{}' from the scope.".format(key))
    return mt_val

  def __setattr__(self, key, value):
    """Update scope data to be functionalized.

    Called for *all* attribute setting.
    """
    if key in ["_active_ranges", "_mutable_state", "_count_subtraces"]:
      object.__setattr__(self, key, value)
    else:
      if self._active_ranges and key not in self._mutable_state:
        raise ValueError(
          "New mutable state '{}' cannot be created inside a loop.".format(key))
      self._mutable_state[key] = value

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    try:
      if exc_type is None:
        if self._active_ranges:  # We have some ranges that we did not exit properly
          self._error_premature_exit_range()
        return True
      else:
        # The exception may come from inside one or more ranges. We let the current
        # exception propagate, assuming it terminates the tracing. If not, the
        # tracers may be left in an inconsistent state.
        return False  # re-raise
    finally:
      # Ensure we leave the global trace_state as we found it
      while self._count_subtraces > 0:
        self.end_subtrace()

  def start_subtrace(self):
    """Starts a nested trace, returns the Trace object."""
    # TODO: This follows the __enter__ part of core.new_master.
    level = core.trace_state.trace_stack.next_level(False)
    master = core.MasterTrace(level, pe.JaxprTrace)
    core.trace_state.trace_stack.push(master, False)
    self._count_subtraces += 1
    return pe.JaxprTrace(master, core.cur_sublevel())

  def end_subtrace(self):
    # TODO: This follows the __exit__ part of core.new_master
    core.trace_state.trace_stack.pop(False)
    self._count_subtraces -= 1


class _BodyTracer(object):
  """Traces the body of the loop and builds a functional control-flow representation.

  This class is also an iterator, only the first iteration is traced.
  """

  def __init__(self, scope, loop_builder):
    """
    Params:
      scope: the current scope
      loop_builder: instance of _LoopBuilder
    """
    self.scope = scope
    self.loop_builder = loop_builder
    self.first_iteration = True  # If we are tracing the first iteration
    # Stack trace, without this line and the s.range function
    self.stack = traceback.StackSummary.from_list(
      cast(List[Any], traceback.extract_stack()[:-2]))

    # Next are state kept from the start of the first iteration to the end of the iteration.
    self.carried_state_initial = {}
    # The parameters that were created for state upon entering an arbitrary iteration.
    self.carried_state_vars = {}

    self.trace = None
    # List of scope fields carried through the loop
    self.carried_state_names = None
    self.init_tree = None  # The PyTreeDef corresponding to carried_state_names
    self.init_vals = None  # The values corresponding to self.init_tree

  def location(self):
    """A multiline string representing the source location of the range."""
    if self.stack is not None:
      return "   ".join(self.stack.format())
    else:
      return ""

  def __iter__(self):
    """Called before starting the first iteration."""
    self.first_iteration = True  # In case we reuse the range
    return self

  def __next__(self):
    if self.first_iteration:
      self.first_iteration = False
      self.scope._push_range(self)
      self.start_tracing_body()
      return self._index_var
    else:
      self.end_tracing_body()
      self.scope._pop_range(self)
      raise StopIteration  # Trace only one iteration.

  def next(self):  # For PY2
    return self.__next__()

  def start_tracing_body(self):
    """Called upon starting the tracing of the loop body."""
    # Make a copy of the current value of the mutable state
    self.carried_state_initial = copy.copy(self.scope._mutable_state)
    # The entire state is carried.
    self.carried_state_names = sorted(self.scope._mutable_state.keys())

    # TODO: This is the first part of partial_eval.trace_to_subjaxpr. Share.
    self.trace = self.scope.start_subtrace()
    # Set the scope._mutable_state to new tracing variables.
    for key, initial in self.carried_state_initial.items():
      mt_aval = _BodyTracer.abstractify(initial)
      mt_pval = pe.PartialVal.unknown(mt_aval)
      mt_var = self.trace.new_arg(mt_pval)
      self.carried_state_vars[key] = mt_var
      self.scope._mutable_state[key] = mt_var

    index_var_aval = _BodyTracer.abstractify(0)
    index_var_pval = pe.PartialVal.unknown(index_var_aval)
    self._index_var = self.trace.new_arg(index_var_pval)

  def end_tracing_body(self):
    """Called when we are done tracing one iteration of the body."""
    # We will turn the body of the loop into a function that takes some values
    # for the scope state (carried_state_names) and returns the values for the
    # same state fields after one execution of the body. For some of the ranges,
    # e.g., scope.range, the function will also take the index_var as last parameter.
    in_tracers = [self.carried_state_vars[ms] for ms in self.carried_state_names]
    if self.loop_builder.can_use_index_var():
      in_tracers += [self._index_var]

    # Make the jaxpr for the body of the loop
    # TODO: See which mutable state was changed in the one iteration.
    # For now, we assume all state changes.
    body_out_tracers = tuple([self.scope._mutable_state[ms]
                              for ms in self.carried_state_names])
    try:
      # If the body actually uses the index variable, and is not allowed to
      # (e.g., cond_range and while_range), then in_tracers will not contain
      # the tracer for the index_var, and trace_to_jaxpr_finalize will throw
      # an assertion error.
      body_typed_jaxpr, body_const_vals = _BodyTracer.trace_to_jaxpr_finalize(
        in_tracers=in_tracers,
        out_tracers=body_out_tracers,
        trace=self.trace)
    except core.UnexpectedTracerError as e:
      if "Tracer not among input tracers" in str(e):
        raise ValueError("Body of cond_range or while_range should not use the "
                         "index variable returned by iterator.") from e
      raise
    # End the subtrace for the loop body, before we trace the condition
    self.scope.end_subtrace()

    carried_init_val = tuple([self.carried_state_initial[ms]
                              for ms in self.carried_state_names])
    carried_init_vals, carried_tree = tree_util.tree_flatten(carried_init_val)

    carried_out_vals = self.loop_builder.build_output_vals(
      self.scope, self.carried_state_names, carried_tree,
      carried_init_vals, body_typed_jaxpr, body_const_vals)
    carried_mutable_state_unflattened = tree_util.tree_unflatten(carried_tree,
                                                                 carried_out_vals)

    # Update the mutable state with the values of the changed vars, after the loop.
    for ms, mv in zip(self.carried_state_names, carried_mutable_state_unflattened):
      self.scope._mutable_state[ms] = mv

  @staticmethod
  def abstractify(x):
    return abstract_arrays.raise_to_shaped(core.get_aval(x))

  @staticmethod
  def trace_to_jaxpr_finalize(in_tracers, out_tracers, trace, instantiate=True):
    # TODO: This is the final part of the partial_eval.trace_to_subjaxpr. Share.
    instantiate = [instantiate] * len(out_tracers)
    out_tracers = safe_map(trace.full_raise, safe_map(core.full_lower, out_tracers))
    out_tracers = safe_map(partial(pe.instantiate_const_at, trace),
                           instantiate, out_tracers)
    jaxpr, consts, env = pe.tracers_to_jaxpr(in_tracers, out_tracers)
    out_pvals = [t.pval for t in out_tracers]
    # TODO: this is from partial_eval.trace_to_jaxpr. Share.
    assert not env

    # TODO: this is from the final part of lax_control_flow._initial_style_jaxpr
    out_avals = safe_map(abstract_arrays.raise_to_shaped, unzip2(out_pvals)[0])
    const_avals = tuple(abstract_arrays.raise_to_shaped(core.get_aval(c))
                        for c in consts)

    in_pvals = [t.pval for t in in_tracers]
    in_avals = tuple(safe_map(abstract_arrays.raise_to_shaped, unzip2(in_pvals)[0]))

    typed_jaxpr = core.TypedJaxpr(pe.convert_constvars_jaxpr(jaxpr),
                                  (), const_avals + in_avals, out_avals)
    return typed_jaxpr, consts


class _LoopBuilder(object):
  """Abstract superclass for the loop builders"""

  def can_use_index_var(self):
    """Whether this kind of loop can use the index var returned by the range iterator."""
    raise NotImplementedError

  def build_output_vals(self, scope, carried_state_names, carried_tree,
                        init_vals, body_typed_jaxpr, body_const_vals):
    """Builds the output values for the loop carried state.

    Params:
      scope: the current Scope object.
      carried_state_names: the list of names of mutable state fields that is
        carried through the body.
      carried_tree: the PyTreeDef for the tuple of carried_state_names.
      init_vals: the initial values on body entry corresponding to the init_tree.
      body_typed_jaxpr: the Jaxpr for the body returning the new values of
        carried_state_names.
      body_const_vals: the constant values for the body.

    Returns:
      the output tracer corresponding to the lax primitive representing the loop.
    """
    raise NotImplementedError

  def __str__(self):
    raise NotImplementedError


class _BoundedLoopBuilder(_LoopBuilder):
  """Builds a lax operation corresponding to a bounded range iteration."""

  def __init__(self, start, stop, step):
    self.start = start
    self.stop = stop
    self.step = step
    self._index_var = None  # The parameter for the index variable

  def can_use_index_var(self):
    return True

  def build_output_vals(self, scope, carried_state_names, carried_tree,
                        init_vals, body_typed_jaxpr, body_const_vals):
    arange_val = jnp.arange(self.start, stop=self.stop, step=self.step)
    return lax_control_flow.scan_p.bind(*itertools.chain(body_const_vals,
                                                         init_vals, [arange_val]),
                                        reverse=False, length=arange_val.shape[0],
                                        jaxpr=body_typed_jaxpr,
                                        num_consts=len(body_const_vals),
                                        num_carry=len(init_vals),
                                        linear=(False,) * (len(body_const_vals) +
                                                           len(init_vals) + 1))


class _CondBuilder(_LoopBuilder):
  """Builds a lax.cond operation."""

  def __init__(self, pred):
    self.pred = pred

  def can_use_index_var(self):
    return False

  def build_output_vals(self, scope, carried_state_names, carried_tree,
                        init_vals, body_typed_jaxpr, body_const_vals):
    # Simulate a pass-through false branch
    init_avals = safe_map(_BodyTracer.abstractify, init_vals)
    false_body_typed_jaxpr, false_body_const_vals, _ = (
      lax_control_flow._initial_style_jaxpr(lambda *args: args,
                                            carried_tree,
                                            tuple(init_avals)))
    args = list(itertools.chain(body_const_vals, init_vals,
                                false_body_const_vals, init_vals))
    return lax_control_flow.cond_p.bind(
        self.pred, *args,
        true_jaxpr=body_typed_jaxpr,
        false_jaxpr=false_body_typed_jaxpr,
        linear=(False,) * len(args))


class _WhileBuilder(_LoopBuilder):
  """Builds a lax.while operation."""

  def __init__(self, cond_func):
    self.cond_func = cond_func  # Function with 0 arguments (can reference the scope)

  def can_use_index_var(self):
    return False

  def build_output_vals(self, scope, carried_state_names, carried_tree,
                        init_vals, body_typed_jaxpr, body_const_vals):
    # Trace the conditional function. cond_func takes 0 arguments, but
    # for lax.while we need a conditional function that takes the
    # carried_state_names. _initial_style_jaxpr will start its own trace and
    # will create tracers for all the carried state. We must put these values
    # in the scope._mutable_state before we trace the conditional
    # function.
    def cond_func_wrapped(*args):
      assert len(args) == len(carried_state_names)
      for ms, init_ms in zip(carried_state_names, args):
        scope._mutable_state[ms] = init_ms
      res = self.cond_func()
      # Conditional function is not allowed to modify the scope state
      for ms, init_ms in zip(carried_state_names, args):
        if not (scope._mutable_state[ms] is init_ms):
          msg = "Conditional function modifies scope.{} field."
          raise ValueError(msg.format(ms))
      return res

    init_avals = safe_map(_BodyTracer.abstractify, init_vals)
    cond_jaxpr, cond_consts, cond_tree = (
      lax_control_flow._initial_style_jaxpr(cond_func_wrapped,
                                            carried_tree,
                                            tuple(init_avals)))
    # TODO: share these checks with lax_control_flow.while
    if not tree_util.treedef_is_leaf(cond_tree):
      msg = "cond_fun must return a boolean scalar, but got pytree {}."
      raise TypeError(msg.format(cond_tree))
    if cond_jaxpr.out_avals != [abstract_arrays.ShapedArray((), onp.bool_)]:
      msg = "cond_fun must return a boolean scalar, but got output type(s) {}."
      raise TypeError(msg.format(cond_jaxpr.out_avals))

    return lax_control_flow.while_p.bind(*itertools.chain(cond_consts,
                                                          body_const_vals,
                                                          init_vals),
                                         cond_nconsts=len(cond_consts),
                                         cond_jaxpr=cond_jaxpr,
                                         body_nconsts=len(body_const_vals),
                                         body_jaxpr=body_typed_jaxpr)
