# Copyright 2025 The JAX Authors.
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

"""Utilities for pull block specs through a fusion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import enum
import functools
import threading
from typing import Any, Protocol

import jax
from jax import lax
from jax._src import ad_util
from jax._src import core
from jax._src import custom_derivatives
from jax._src import pjit
from jax._src import prng
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.fuser import fuser_utils
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
import jax.numpy as jnp
import numpy as np

# TODO(sharadmv): Enable type checking.
# mypy: ignore-errors

pull_block_spec_rules: dict[core.Primitive, PullBlockSpecRuleFn] = {}


@dataclasses.dataclass
class PullRuleContext:
  avals_in: tuple[core.AbstractValue, ...]
  avals_out: tuple[core.AbstractValue, ...]
  out_usages: tuple[set[Usage], ...]
  eval_function: Any = dataclasses.field(default=None, init=False)
  scalar_prefetch_fn: Any = dataclasses.field(default=None, init=False)
  scalar_prefetch_handler: Any | None
  grid: tuple[int | jax.Array, ...] | None

  def __post_init__(self):
    self._scalar_prefetch = None

  def set_eval_function(self, eval_function):
    self.eval_function = eval_function
    return eval_function


@dataclasses.dataclass
class PushRuleContext:
  avals_in: tuple[core.AbstractValue, ...]
  avals_out: tuple[core.AbstractValue, ...]


def make_scalar_prefetch_handler(*args):
  def scalar_prefetch_getter(*sp_inputs):
    result = sp_inputs
    for i in args:
      result = result[i]
    return result

  return scalar_prefetch_getter


def _block_size(dim: pallas_core.Element | int | None) -> int | None:
  match dim:
    case (
        pallas_core.Element()
        | pallas_core.BoundedSlice()
        | pallas_core.Blocked()
    ):
      return dim.block_size
    case pallas_core.Squeezed() | None:
      return None
    case _:
      return dim  # pytype: disable=bad-return-type


@dataclasses.dataclass
class UsageRuleContext:
  avals_in: tuple[core.AbstractValue, ...]
  avals_out: tuple[core.AbstractValue, ...]


def compute_usage(jaxpr: core.Jaxpr, jaxpr_out_usages):
  # TODO(sharadmv): maybe simplify this by only identifying scalar prefetch
  # and then DCEing it out.
  usage_env: dict[core.Atom, set[Usage]] = {}

  def read_usage_env(atom: core.Atom) -> set[Usage]:
    assert not isinstance(atom, core.Literal)
    return usage_env.get(atom, {Usage.REGULAR})

  def write_usage_env(atom: core.Atom, usage: set[Usage]):
    if isinstance(atom, core.Literal):
      return
    if atom not in usage_env:
      usage_env[atom] = set()
    usage_env[atom] |= usage

  util.safe_map(write_usage_env, jaxpr.outvars, jaxpr_out_usages)

  for eqn in jaxpr.eqns[::-1]:
    out_usages = util.safe_map(read_usage_env, eqn.outvars)
    rule = usage_rules.get(eqn.primitive, None)
    if rule:
      usage_rule_ctx = UsageRuleContext(
          avals_in=tuple(v.aval for v in eqn.invars),
          avals_out=tuple(v.aval for v in eqn.outvars),
      )
      if eqn.primitive.multiple_results:
        in_usages = rule(usage_rule_ctx, out_usages, **eqn.params)
      else:
        in_usages = rule(usage_rule_ctx, out_usages[0], **eqn.params)
    else:
      # Usages are forwarded
      all_usages = set.union(*out_usages)
      in_usages = [all_usages] * len(eqn.invars)
    util.safe_map(write_usage_env, eqn.invars, in_usages)
  return read_usage_env


@dataclasses.dataclass(frozen=True)
class KernelEvalContext:
  scalar_prefetch: Any | None
  program_ids: tuple[int | jax.Array, ...] | None
  avals_in: tuple[core.AbstractValue, ...] | None
  avals_out: tuple[core.AbstractValue, ...] | None
  in_block_specs: tuple[pallas_core.BlockSpec, ...] | None
  out_block_specs: tuple[pallas_core.BlockSpec, ...] | None
  grid: tuple[int | jax.Array, ...] | None
  scalar_prefetch_handler: Any | None
  out_usages: tuple[set[Usage], ...] | None

  def get_program_ids(self):
    if self.program_ids is None:
      raise ValueError('Program ids not available.')
    return self.program_ids

  def get_in_block_indices(self):
    with _sp_context(*self.scalar_prefetch):
      return tuple(
          bs.index_map(*self.program_ids) for bs in self.in_block_specs
      )

  def get_out_block_indices(self):
    with _sp_context(*self.scalar_prefetch):
      return tuple(
          bs.index_map(*self.program_ids) for bs in self.out_block_specs
      )


_illegal = object()


class _SpEnv(threading.local):

  def __init__(self):
    self.scalar_prefetch = None


_sp_env = _SpEnv()


@contextlib.contextmanager
def _sp_context(*scalar_prefetch):
  assert _sp_env.scalar_prefetch is None, scalar_prefetch
  _sp_env.scalar_prefetch = scalar_prefetch
  try:
    yield
  finally:
    _sp_env.scalar_prefetch = None


def _get_scalar_prefetch():
  return _sp_env.scalar_prefetch


def _wrap_block_spec_scalar_prefetch(
    block_spec: pallas_core.BlockSpec,
    num_grid_args: int,
) -> pallas_core.BlockSpec:
  if block_spec is pallas_core.no_block_spec or block_spec.index_map is None:
    return block_spec

  def new_index_map(*args_and_scalar_prefetch):
    args, scalar_prefetch = util.split_list(
        args_and_scalar_prefetch,
        [num_grid_args],
    )
    with _sp_context(*scalar_prefetch):
      return block_spec.index_map(*args)

  return block_spec.replace(index_map=new_index_map)


_unwrap_cache: dict[int, pallas_core.BlockSpec] = {}


def _unwrap_block_spec_scalar_prefetch(
    block_spec: pallas_core.BlockSpec,
) -> pallas_core.BlockSpec:
  if id(block_spec) in _unwrap_cache:
    return _unwrap_cache[id(block_spec)]

  def new_index_map(*args):
    scalar_prefetch = _get_scalar_prefetch()
    assert scalar_prefetch is not None
    return block_spec.index_map(*args, *scalar_prefetch)

  out_block_spec = block_spec.replace(index_map=new_index_map)
  _unwrap_cache[id(block_spec)] = out_block_spec
  return out_block_spec


def pull_block_spec(
    f: Callable,
    out_block_specs: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
    *,
    scalar_prefetch_handler: Any | None = None,
    grid: tuple[int | jax.Array, ...] | None = None,
):
  def wrapped(*args, **kwargs):
    jaxpr, consts, in_tree, out_tree_ = fuser_utils.make_jaxpr(
        f, *args, **kwargs
    )
    del out_tree_
    jaxpr_out_usages = [{Usage.REGULAR}] * len(jaxpr.outvars)
    block_specs_ = jax.tree.map(
        _unwrap_block_spec_scalar_prefetch, out_block_specs
    )
    flat_block_specs, out_tree = jax.tree.flatten(block_specs_)
    jaxpr, used_consts, used_invars = pe.dce_jaxpr_consts(
        jaxpr,
        used_outputs=[True] * len(jaxpr.outvars),
        instantiate=True,
    )
    assert all(used_invars)
    assert all(used_consts)
    read_usage_env = compute_usage(jaxpr, jaxpr_out_usages)
    in_block_specs, env, read_usage_env = _pull_block_spec(
        jaxpr,
        tuple(flat_block_specs),
        scalar_prefetch_handler=scalar_prefetch_handler,
        read_usage_env=read_usage_env,
        grid=grid,
    )
    kernel_fn = make_kernel_function(
        jaxpr,
        consts,
        in_tree,
        out_tree,
        read_usage_env,
        in_block_specs,
        env,
        scalar_prefetch_handler,
        grid,
    )
    in_block_specs = jax.tree.unflatten(in_tree, in_block_specs)
    in_block_specs = jax.tree.map(
        functools.partial(
            _wrap_block_spec_scalar_prefetch,
            num_grid_args=len(grid),
        ),
        in_block_specs,
    )
    in_block_arg_specs, in_block_kwarg_specs = in_block_specs
    return kernel_fn, in_block_arg_specs, in_block_kwarg_specs

  return wrapped


def _pull_block_spec(
    jaxpr: core.Jaxpr,
    out_block_specs: tuple[pallas_core.BlockSpec, ...],
    *,
    read_usage_env: Callable[[core.Var], set[Usage]],
    scalar_prefetch_handler: Any | None = None,
    grid: tuple[int | jax.Array, ...],
) -> tuple[
    tuple[pallas_core.BlockSpec | pallas_core.NoBlockSpec, ...],
    tuple[dict[core.Var, pallas_core.BlockSpec], dict[int, Any]],
    Any,
]:
  jaxpr_invar_usages = util.safe_map(read_usage_env, jaxpr.invars)
  env: dict[core.Var, pallas_core.BlockSpec] = {}
  scalar_prefetch_fn_env = {}

  for outvar, bs in zip(jaxpr.outvars, out_block_specs, strict=True):
    assert isinstance(outvar, core.Var)
    env[outvar] = bs

  def _read_block_spec(atom: core.Atom) -> pallas_core.BlockSpec | Any:
    if isinstance(atom, core.Literal):
      return pallas_core.no_block_spec
    return env.get(atom, pallas_core.no_block_spec)

  def _write_block_spec(atom: core.Atom, block_spec: pallas_core.BlockSpec):
    if isinstance(atom, core.Literal):
      return
    env[atom] = block_spec

  for i, eqn in reversed(list(enumerate(jaxpr.eqns))):
    eqn_out_block_specs = tuple(util.safe_map(_read_block_spec, eqn.outvars))
    if all(bs is pallas_core.no_block_spec for bs in eqn_out_block_specs):
      continue
    rule = pull_block_spec_rules.get(eqn.primitive, None)
    if not rule:
      raise NotImplementedError(eqn.primitive, eqn_out_block_specs)
    ctx = PullRuleContext(
        avals_in=tuple(v.aval for v in eqn.invars),
        avals_out=tuple(v.aval for v in eqn.outvars),
        out_usages=tuple(read_usage_env(v) for v in jaxpr.outvars),
        scalar_prefetch_handler=scalar_prefetch_handler,
        grid=grid,
    )
    if eqn.primitive.multiple_results:
      in_block_specs = rule(ctx, eqn_out_block_specs, **eqn.params)
    else:
      in_block_specs = rule(ctx, eqn_out_block_specs[0], **eqn.params)

    eqn_invar_usages = [
        read_usage_env(v) if not isinstance(v, core.Literal) else set()
        for v in eqn.invars
    ]
    if any(Usage.SCALAR_PREFETCH in usage for usage in eqn_invar_usages):
      scalar_prefetch_vars = [
          Usage.SCALAR_PREFETCH in usage for usage in eqn_invar_usages
      ]
      needed_invars = [
          v
          for v, sp in zip(eqn.invars, scalar_prefetch_vars)
          if sp or isinstance(v, core.Literal)
      ]
      scalar_prefetch_jaxpr_no_dce = core.Jaxpr(
          jaxpr.constvars,
          jaxpr.invars,
          needed_invars,
          jaxpr.eqns[: jaxpr.eqns.index(eqn)],
          debug_info=jaxpr.debug_info,
      )
      scalar_prefetch_jaxpr, used_consts, used_invars = pe.dce_jaxpr_consts(
          scalar_prefetch_jaxpr_no_dce,
          [True] * len(scalar_prefetch_jaxpr_no_dce.outvars),
      )
      assert not any(used_invars)
      scalar_prefetch_jaxpr = scalar_prefetch_jaxpr.replace(
          constvars=[],
          invars=jaxpr.constvars,
      )

      def _scalar_prefetch_fn(jaxpr):
        if grid is None:
          raise ValueError('Grid must be provided to pull_block_spec.')
        args = scalar_prefetch_handler(*_get_scalar_prefetch())
        # Load from SMEM
        args = [a[0] for a in args]
        return core.eval_jaxpr(jaxpr, [], *args)

      scalar_prefetch_fn = functools.partial(
          _scalar_prefetch_fn, scalar_prefetch_jaxpr
      )
      ctx.scalar_prefetch_fn = scalar_prefetch_fn_env[i] = scalar_prefetch_fn
    for v, in_block_spec in zip(eqn.invars, in_block_specs, strict=True):
      # TODO(cjfj): Check that index map functions are equivalent (in jaxpr).
      if (
          not isinstance(v, core.Literal)
          and v in env
          and env[v].block_shape != in_block_spec.block_shape
      ):
        in_block_spec = pallas_core.BlockSpec(_illegal, _illegal)  # pytype: disable=wrong-arg-types
      _write_block_spec(v, in_block_spec)

  def _get_in_block_spec(v, usage):
    if usage == {Usage.SCALAR_PREFETCH}:
      return None
    bs = env.get(v, pallas_core.no_block_spec)
    if bs is not pallas_core.no_block_spec:
      if bs.index_map is _illegal:  # pytype: disable=attribute-error
        raise ValueError(f'Found cycle:\n{jaxpr}')
    return bs

  in_block_specs = tuple(
      _get_in_block_spec(v, usage)
      for v, usage in zip(jaxpr.invars, jaxpr_invar_usages)
  )
  return (
      tuple(in_block_specs),
      (env, scalar_prefetch_fn_env),
      read_usage_env,
  )


def make_kernel_function(
    jaxpr: core.Jaxpr,
    consts,
    in_tree,
    out_tree,
    read_usage_env,
    in_block_specs,
    block_spec_env,
    scalar_prefetch_handler,
    grid,
):
  in_avals = [v.aval for v in jaxpr.invars]
  invar_usages = util.safe_map(read_usage_env, jaxpr.invars)
  bs_env, scalar_prefetch_fn_env = block_spec_env

  def _remove_nones(
      shape: tuple[pallas_core.BlockDim | int | None, ...] | None,
  ) -> tuple[int, ...]:
    assert shape is not None
    new_shape = tuple(_block_size(s) for s in shape)
    return tuple(s for s in new_shape if s is not None)

  _no_aval = object()

  def _get_block_aval(bs, aval):
    if isinstance(aval, state.AbstractRef):
      return aval
    if bs is pallas_core.no_block_spec or bs is None:
      return _no_aval
    if bs.block_shape is None:
      return aval
    return aval.update(shape=_remove_nones(bs.block_shape))  # pytype: disable=attribute-error

  in_block_avals = [
      _get_block_aval(bs, aval)
      for aval, bs in zip(in_avals, in_block_specs, strict=True)
  ]
  unflat_in_block_arg_avals, unflat_in_block_kwarg_avals = (
      tree_util.tree_unflatten(in_tree, in_block_avals)
  )
  unflat_arg_usages, unflat_kwarg_usages = tree_util.tree_unflatten(
      in_tree, invar_usages
  )

  def sds_like(x):
    if x is _no_aval:
      return _no_aval
    return jax.ShapeDtypeStruct(x.shape, x.dtype)

  kernel_in_type = jax.tree.map(
      sds_like, (unflat_in_block_arg_avals, unflat_in_block_kwarg_avals)
  )

  def _read_block_spec(atom: core.Atom) -> pallas_core.BlockSpec | Any:
    if isinstance(atom, core.Literal):
      return pallas_core.no_block_spec
    return bs_env.get(atom, pallas_core.no_block_spec)

  def kernel_fn(program_ids, scalar_prefetch, *args, **kwargs):
    def _check_args(prefix, path, x, y, usage):
      if usage == {Usage.SCALAR_PREFETCH}:
        return
      if y is _no_aval:
        return
      x_aval, y_aval = core.get_aval(x), core.get_aval(y)
      if x_aval.shape != y_aval.shape:
        raise ValueError(
            f'Shapes do not match: actual={x_aval.shape} !='
            f' expected={y_aval.shape}. Path:'
            f' {prefix}{jax.tree_util.keystr(path)}. Expected type:'
            f' {kernel_in_type}. Actual args: {(args, kwargs)}'
        )
      if x_aval.dtype != y_aval.dtype:
        raise ValueError(
            f'DTypes do not match: actual={x_aval.dtype} !='
            f' expected={y_aval.dtype}. Path:'
            f' {prefix}{jax.tree_util.keystr(path)}. Expected type:'
            f' {kernel_in_type}. Actual args: {(args, kwargs)}'
        )

    jax.tree_util.tree_map_with_path(
        functools.partial(_check_args, 'args'),
        args,
        kernel_in_type[0],
        unflat_arg_usages,
    )
    jax.tree_util.tree_map_with_path(
        functools.partial(_check_args, 'kwargs'),
        kwargs,
        kernel_in_type[1],
        unflat_kwarg_usages,
    )
    flat_args, in_tree_ = tree_util.tree_flatten((args, kwargs))
    if in_tree_ != tree_util.tree_structure(kernel_in_type):
      raise ValueError(f'Expected {kernel_in_type} PyTree, got {in_tree_}')
    env = {}

    def read_env(atom):
      match atom:
        case core.Literal():
          return atom.val
        case core.Var():
          return env.get(atom, None)

    def write_env(var, val):
      env[var] = val

    for const, constvar in zip(consts, jaxpr.constvars):
      env[constvar] = const
    for invar, arg, usage in zip(jaxpr.invars, flat_args, invar_usages):
      if Usage.REGULAR in usage:
        env[invar] = arg
    for i, eqn in enumerate(jaxpr.eqns):
      outvar_usages = [
          read_usage_env(v) if not isinstance(v, core.Literal) else set()
          for v in eqn.outvars
      ]
      if any(Usage.REGULAR in usage for usage in outvar_usages):
        in_vals = util.safe_map(read_env, eqn.invars)
        # TODO(sharadmv,justinfu): preserve source mapping
        if not (eval_rule := eval_rules.get(eqn.primitive, None)):
          raise NotImplementedError(eqn.primitive)
        out_block_specs = tuple(util.safe_map(_read_block_spec, eqn.outvars))
        in_block_specs = tuple(util.safe_map(_read_block_spec, eqn.invars))
        out_usages = tuple(read_usage_env(v) for v in eqn.outvars)
        # We need to substitute in scalar prefetch values into equation invars.
        if scalar_prefetch_fn := scalar_prefetch_fn_env.get(i, None):
          # Evaluate scalar prefetch function.
          with _sp_context(*scalar_prefetch):
            scalar_prefetch_vals = scalar_prefetch_fn()
          # Some results of the SP function will be literals.
          eqn_invar_usages = [
              read_usage_env(v)
              if not isinstance(v, core.Literal)
              else {Usage.SCALAR_PREFETCH}
              for v in eqn.invars
          ]
          sp_iter = iter(scalar_prefetch_vals)
          for i, usage in enumerate(eqn_invar_usages):
            if usage == {Usage.SCALAR_PREFETCH}:
              if not isinstance(eqn.invars[i], core.Literal):
                in_vals[i] = next(sp_iter)
        eval_ctx = KernelEvalContext(
            avals_in=tuple(v.aval for v in eqn.invars),
            avals_out=tuple(v.aval for v in eqn.outvars),
            scalar_prefetch=scalar_prefetch,
            program_ids=tuple(program_ids),
            in_block_specs=in_block_specs,
            out_block_specs=out_block_specs,
            scalar_prefetch_handler=scalar_prefetch_handler,
            grid=grid,
            out_usages=out_usages,
        )
        outs = eval_rule(eval_ctx, *in_vals, **eqn.params)
        if not eqn.primitive.multiple_results:
          outs = [outs]
        util.safe_map(write_env, eqn.outvars, outs)
    out = util.safe_map(read_env, jaxpr.outvars)
    return tree_util.tree_unflatten(out_tree, out)

  return kernel_fn


def get_fusion_values(
    fusion: Callable, *args, **kwargs
) -> tuple[Callable, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  jaxpr, values, in_tree, out_tree = fuser_utils.make_jaxpr(
      fusion, *args, **kwargs
  )
  assert len(values) == len(jaxpr.constvars), (jaxpr, values)
  out_usages = tuple({Usage.REGULAR} for _ in jaxpr.outvars)
  read_usage_env = compute_usage(jaxpr, out_usages)
  constvar_usages = util.safe_map(read_usage_env, jaxpr.constvars)
  invar_usages = util.safe_map(read_usage_env, jaxpr.invars)
  del invar_usages  # These don't correspond to values
  # Add leading dimension to scalar prefetch values so Mosaic won't be upset.
  is_scalar_prefetch = tuple(
      Usage.SCALAR_PREFETCH in usage for usage in constvar_usages
  )
  regular_values, scalar_prefetch_values = util.partition_list(
      is_scalar_prefetch, values
  )
  # scalar_prefetch_values = [x for x in scalar_prefetch_values]

  def new_kernel_fn(values, *args, **kwargs):
    values = util.merge_lists(
        is_scalar_prefetch, values, scalar_prefetch_values
    )
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    out_flat = core.eval_jaxpr(jaxpr, values, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return new_kernel_fn, tuple(regular_values), tuple(scalar_prefetch_values)


# # Interpreter rules


# ## Usage interpreter rules
class Usage(enum.Enum):
  REGULAR = 0
  SCALAR_PREFETCH = 1


class UsageRuleFn(Protocol):

  def __call__(
      self,
      ctx: UsageRuleContext,
      used_outs: Sequence[set[Usage]] | set[Usage],
      **params: Any,
  ) -> Sequence[set[Usage]]:
    ...


usage_rules: dict[core.Primitive, UsageRuleFn] = {}


def register_usage_rule(
    prim: core.Primitive,
) -> Callable[[UsageRuleFn], UsageRuleFn]:

  def wrapper(
      f: UsageRuleFn,
  ) -> UsageRuleFn:
    usage_rules[prim] = f
    return f

  return wrapper


# ## Eval interpreter rules


class EvalRuleFn(Protocol):

  def __call__(
      self,
      ctx: KernelEvalContext,
      *args: Any,
      **params: Any,
  ) -> Sequence[Any]:
    ...


eval_rules: dict[core.Primitive, EvalRuleFn] = {}


def register_eval_rule(
    prim: core.Primitive,
) -> Callable[[EvalRuleFn], EvalRuleFn]:
  def wrapper(
      f: EvalRuleFn,
  ) -> EvalRuleFn:
    eval_rules[prim] = f
    return f

  return wrapper


# ## Pull block spec interpreter rules


class PullBlockSpecRuleFn(Protocol):

  def __call__(
      self,
      ctx: PullRuleContext,
      block_spec: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
      **params: Any,
  ) -> Sequence[pallas_core.BlockSpec]:
    ...


def register_pull_block_spec_rule(
    prim: core.Primitive,
) -> Callable[[PullBlockSpecRuleFn], PullBlockSpecRuleFn]:

  def wrapper(
      f: PullBlockSpecRuleFn,
  ) -> PullBlockSpecRuleFn:
    pull_block_spec_rules[prim] = f
    return f

  return wrapper


# Primitive rule implementations


def _eltwise_eval_rule(prim, ctx, x, **params):
  del ctx
  return prim.bind(x, **params)


def _eltwise_pull_rule(
    prim: core.Primitive,
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    **params,
) -> Sequence[pallas_core.BlockSpec]:
  del prim, ctx, params
  return [block_spec]


def _eltwise_usage_rule(
    prim: core.Primitive, ctx: UsageRuleContext, used_out: set[Usage], **params
) -> Sequence[set[Usage]]:
  del ctx, prim, params
  return [used_out]


def _bcast_block_spec(
    block_spec: pallas_core.BlockSpec, i: int
) -> pallas_core.BlockSpec:
  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    assert len(idx) == len(block_spec.block_shape)
    idx = util.tuple_update(idx, i, 0)
    return idx

  if block_spec.block_shape[i] is None:
    return pallas_core.BlockSpec(block_spec.block_shape, new_index_map)

  # TODO(wdvi): This is a hack needed since lowering rules require block shape
  # to contain either all pl.Element or none
  bcast_dim_block_shape = 1
  if isinstance(block_spec.block_shape[i], pallas_core.Element):
    bcast_dim_block_shape = pallas_core.Element(1)
  new_block_shape = util.tuple_update(  # pytype: disable=wrong-arg-types
      block_spec.block_shape, i, bcast_dim_block_shape
  )
  return pallas_core.BlockSpec(new_block_shape, new_index_map)


def _binop_usage_rule(prim, ctx, used_out: set[Usage]):
  del prim
  if used_out == {Usage.SCALAR_PREFETCH}:
    return [{Usage.SCALAR_PREFETCH}, {Usage.SCALAR_PREFETCH}]
  elif used_out == {Usage.REGULAR}:
    usage = [{Usage.REGULAR} for _ in ctx.avals_in]
    return usage
  else:
    return [set()] * len(ctx.avals_in)


def _binop_eval_rule(prim, ctx, x, y):
  del ctx
  return prim.bind(x, y)


def _binop_pull_rule(prim, ctx: PullRuleContext, block_spec):
  l_block_spec = block_spec
  r_block_spec = block_spec
  left_aval, right_aval = ctx.avals_in
  assert isinstance(left_aval, core.ShapedArray)
  assert isinstance(right_aval, core.ShapedArray)

  @ctx.set_eval_function
  def _eval_function(_, x, y):
    sp_index = 0
    if x is None:
      x = ctx.scalar_prefetch_fn()[sp_index]
      sp_index += 1
    if y is None:
      y = ctx.scalar_prefetch_fn()[sp_index]
    return prim.bind(x, y)

  if not right_aval.shape:
    return [block_spec, pallas_core.no_block_spec]
  if not left_aval.shape:
    return [pallas_core.no_block_spec, block_spec]
  for i, (l, r) in enumerate(
      zip(left_aval.shape, right_aval.shape, strict=True)
  ):
    if l == 1 and r != 1:
      l_block_spec = _bcast_block_spec(l_block_spec, i)
    if r == 1 and l != 1:
      r_block_spec = _bcast_block_spec(r_block_spec, i)

  return [l_block_spec, r_block_spec]


def register_default_eval_rule(prim: core.Primitive):
  def default_rule(ctx, *args, **params):
    assert all(bs is pallas_core.no_block_spec for bs in ctx.out_block_specs)
    return prim.bind(*args, **params)

  register_eval_rule(prim)(default_rule)


def register_binop_rule(prim: core.Primitive):
  register_pull_block_spec_rule(prim)(functools.partial(_binop_pull_rule, prim))
  register_usage_rule(prim)(functools.partial(_binop_usage_rule, prim))
  register_eval_rule(prim)(functools.partial(_binop_eval_rule, prim))


register_default_eval_rule(state_primitives.get_p)

register_binop_rule(lax.mul_p)
register_binop_rule(lax.add_p)
register_binop_rule(lax.sub_p)
register_binop_rule(lax.div_p)
register_binop_rule(lax.max_p)
register_binop_rule(lax.lt_p)
register_binop_rule(lax.le_p)
register_binop_rule(lax.eq_p)
register_binop_rule(lax.gt_p)
register_binop_rule(lax.ge_p)
register_binop_rule(lax.or_p)
register_binop_rule(lax.xor_p)
register_binop_rule(lax.and_p)
register_binop_rule(lax.shift_right_logical_p)
register_binop_rule(ad_util.add_any_p)
register_binop_rule(lax.pow_p)


@register_eval_rule(lax.select_n_p)
def _select_n_eval_rule(ctx: KernelEvalContext, *args):
  return jax.lax.select_n(*args)


@register_pull_block_spec_rule(lax.select_n_p)
def _select_n_pull_block_spec_rule(
    ctx: PullRuleContext, block_spec: pallas_core.BlockSpec
) -> Sequence[pallas_core.BlockSpec]:
  in_aval = ctx.avals_in[0]
  assert isinstance(in_aval, core.ShapedArray)
  if in_aval.shape:
    return [block_spec] * len(ctx.avals_in)
  return [pallas_core.no_block_spec, *[block_spec] * (len(ctx.avals_in) - 1)]


@register_eval_rule(lax.squeeze_p)
def _squeeze_eval_rule(ctx: KernelEvalContext, x: jax.Array, **params: Any):
  del ctx, params
  return x


@register_pull_block_spec_rule(lax.squeeze_p)
def _squeeze_block_spec(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dimensions: tuple[int, ...],
) -> Sequence[pallas_core.BlockSpec]:
  del ctx
  if block_spec is pallas_core.no_block_spec:
    return [pallas_core.no_block_spec]

  def new_index_map(*args):
    idx = tuple(block_spec.index_map(*args))
    assert len(idx) == len(block_spec.block_shape)
    for dim in dimensions:
      idx = util.tuple_insert(idx, dim, 0)
    return idx

  new_block_shape = tuple(block_spec.block_shape)
  for dim in dimensions:
    new_block_shape = util.tuple_insert(new_block_shape, dim, None)

  return [pallas_core.BlockSpec(new_block_shape, new_index_map)]


@register_eval_rule(lax.slice_p)
def _slice_eval_rule(ctx, x, **params):
  del params
  out_block_shape = ctx.out_block_specs[0].block_shape
  assert len(x.shape) == sum(
      1
      for bs in out_block_shape
      if not (bs is None or isinstance(bs, pallas_core.Squeezed))
  )
  return x


def _offset_indexer(
    bs: pallas_core.BlockDim | int | None,
    indexer,
    slice_start,
    slice_size,
):
  # Short-circuit if the slice start is just at zero.
  if isinstance(slice_start, int) and slice_start == 0:
    return indexer
  match bs:
    case None | pallas_core.Squeezed():
      return indexer + slice_start
    case pallas_core.Element(block_size):
      _maybe_static_check(
          slice_start % block_size == 0,
          f'slice_start is not a multiple of block_size {block_size}',
      )
      _maybe_static_check(
          slice_size % block_size == 0,
          f'slice_size is not a multiple of block_size {block_size}',
      )
      return indexer + slice_start
    case int() | pallas_core.Blocked():
      block_size = _block_size(bs)
      _maybe_static_check(
          slice_start % block_size == 0,
          f'slice_start is not a multiple of block_size {block_size}',
      )
      _maybe_static_check(
          slice_size % block_size == 0,
          f'slice_size is not a multiple of block_size {block_size}',
      )
      # indexer is a block index so we need to offset it by the block offset.
      return indexer + slice_start // block_size
    case pallas_core.BoundedSlice(block_size):
      assert isinstance(indexer, indexing.Slice)
      _maybe_static_check(
          indexer.start % block_size == 0,
          f'slice_start is not a multiple of block_size {block_size}',
      )
      _maybe_static_check(
          indexer.size % block_size == 0,
          f'slice_size is not a multiple of block_size {block_size}',
      )
      return indexing.ds(indexer.start + slice_start, indexer.size)
    case _:
      raise ValueError(f'Unsupported block size {bs}')


def _maybe_static_check(pred: bool, msg: str):
  # Tries to emit a static error if possible, otherwise falls back to runtime.
  from jax.experimental import checkify

  if isinstance(pred, jax.Array):
    checkify.check(pred, msg, debug=True)
  else:
    if not pred:
      raise ValueError(msg)


@register_pull_block_spec_rule(lax.slice_p)
def _slice_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    start_indices: tuple[int, ...],
    limit_indices: tuple[int, ...],
    strides: tuple[int, ...] | None,
):
  del ctx
  if strides is not None and not all(stride == 1 for stride in strides):
    raise NotImplementedError('strides are not supported yet')
  slice_sizes = tuple(
      int(end - start) for start, end in zip(start_indices, limit_indices)
  )
  # Do some basic checks
  for bs, slice_start, slice_size in zip(
      block_spec.block_shape, start_indices, slice_sizes
  ):
    match bs:
      case None | pallas_core.Squeezed():
        continue
      case pallas_core.BoundedSlice() | pallas_core.Element():
        block_size = _block_size(bs)
        # Require that block_size no bigger than the slice.
        if block_size > slice_size:
          raise ValueError(
              f'Block size {block_size} is larger than the slice size'
              f' {slice_size}'
          )
      case _:
        block_size = _block_size(bs)
        assert slice_start % block_size == 0, (
            start_indices,
            block_spec.block_shape,
        )
        assert slice_size % block_size == 0, (
            slice_sizes,
            block_spec.block_shape,
        )

  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    assert len(idx) == len(block_spec.block_shape)
    idx = tuple(
        _offset_indexer(bs, i, start, size)
        for bs, i, start, size in zip(
            block_spec.block_shape, idx, start_indices, slice_sizes, strict=True
        )
    )
    return idx

  return [pallas_core.BlockSpec(block_spec.block_shape, new_index_map)]


@register_usage_rule(lax.dynamic_slice_p)
def _dynamic_slice_usage_rule(ctx, used_out: set[Usage], **params):
  del params
  if used_out == {Usage.SCALAR_PREFETCH}:
    raise NotImplementedError('scalar prefetch not supported yet')
  elif used_out == {Usage.REGULAR}:
    usage = [used_out] + [{Usage.SCALAR_PREFETCH}] * (len(ctx.avals_in) - 1)
    return usage
  else:
    return [set()] * len(ctx.avals_in)


@register_eval_rule(lax.dynamic_slice_p)
def _dynamic_slice_eval_rule(ctx, x, *args, **params):
  del ctx, params
  return x


@register_pull_block_spec_rule(lax.dynamic_slice_p)
def _dynamic_slice_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    slice_sizes: tuple[int, ...],
):

  def new_index_map(*args):
    slice_starts = ctx.scalar_prefetch_fn()
    if len(slice_starts) != len(block_spec.block_shape):
      raise ValueError(
          f'Expected {len(block_spec.block_shape)} slice starts, got'
          f' {len(slice_starts)}'
      )
    idx = block_spec.index_map(*args)
    assert len(idx) == len(block_spec.block_shape)

    # Once we have the indices, we need to offset them by the dynamic slice
    # indices. The dynamic slice indices index the full array. For example,
    # let's say we have a [l, m, n] array and are provided 3 dynamic slice
    # start indices [i, j, k] with sizes [s_l, s_m, s_n]. To perform the slice,
    # we need to compute the indices of the block that correspond to that slice
    # in the [l, m, n] array. If we have block sizes [b_l, b_m, b_n], we require
    # that i % b_l == 0, j % b_m == 0, k % b_n == 0 and the slice sizes are
    # multiples of the block sizes. The indices of the block that correspond to
    # the slice are then given by (i // b_l, j // b_m, k // b_n).
    # We then add these block indices to block indices produced by the index
    # map
    block_indices = tuple(
        _offset_indexer(s, i, start, size)
        for i, s, start, size in zip(
            idx, block_spec.block_shape, slice_starts, slice_sizes, strict=True
        )
    )
    return block_indices

  new_block_spec = pallas_core.BlockSpec(block_spec.block_shape, new_index_map)
  return [new_block_spec] + [pallas_core.no_block_spec] * (
      len(ctx.avals_in) - 1
  )


@register_pull_block_spec_rule(state_primitives.swap_p)
def _swap_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    **kwargs,
):
  del ctx, kwargs
  # The output and val block spec are the same.
  return [block_spec, block_spec]


@register_eval_rule(state_primitives.swap_p)
def _swap_eval_rule(ctx: KernelEvalContext, ref, val, *idx, tree):
  indexers = tree_util.tree_unflatten(tree, idx)
  ref_aval, _ = ctx.avals_in[:2]
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[2:])
  assert hasattr(ref_aval, 'shape')
  if len(indexers) > 1:
    raise NotImplementedError('swap not supported yet')
  indexer_aval = indexers_avals[0]
  for idx_aval, size in zip(indexer_aval.indices, ref_aval.shape, strict=True):
    if not isinstance(idx_aval, indexing.Slice):
      raise NotImplementedError('swap not supported yet')
    if not isinstance(idx_aval.start, int):
      raise NotImplementedError('swap not supported yet')
    if not isinstance(idx_aval.size, int):
      raise NotImplementedError('swap not supported yet')
    if idx_aval.stride != 1:
      raise NotImplementedError('swap not supported yet')
    if idx_aval.start != 0:
      raise NotImplementedError('swap not supported yet')
    if idx_aval.size != size:
      raise NotImplementedError('swap not supported yet')
  # We have a pure slice so now we can just re-index the ref according to the
  # block indices.
  block_spec = ctx.out_block_specs[0]
  block_idx = ctx.get_out_block_indices()[0]

  def _slice(i, b):
    if not isinstance(b, int):
      raise NotImplementedError('swap not supported yet')
    return i if b is None else indexing.ds(i * b, b)

  indexer = tuple(
      _slice(i, b)
      for i, b in zip(block_idx, block_spec.block_shape, strict=True)
  )
  return ref.swap(val, idx=indexer)


@register_pull_block_spec_rule(state_primitives.get_p)
def _get_pull_rule(
    ctx: PullRuleContext, block_spec: pallas_core.BlockSpec, *, tree
):
  ref_aval = ctx.avals_in[0]
  assert hasattr(ref_aval, 'shape')
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[1:])
  if len(indexers_avals) > 1:
    raise NotImplementedError('get not supported yet')
  indexer_aval = indexers_avals[0]
  block_shape_iter = iter(block_spec.block_shape)
  block_shape = []
  if not all(
      isinstance(bd, (int, pallas_core.Blocked, pallas_core.Squeezed, None))
      for bd in block_spec.block_shape
  ):
    raise NotImplementedError('get not supported yet')
  for idx_aval, size in zip(indexer_aval.indices, ref_aval.shape, strict=True):
    if not isinstance(idx_aval, indexing.Slice):
      assert hasattr(idx_aval, 'shape') and not idx_aval.shape
      block_shape.append(pallas_core.Squeezed())
      continue
    if not isinstance(idx_aval.start, int):
      raise NotImplementedError('get not supported yet')
    if not isinstance(idx_aval.size, int):
      raise NotImplementedError('get not supported yet')
    if idx_aval.stride != 1:
      raise NotImplementedError('get not supported yet')
    if idx_aval.start != 0:
      raise NotImplementedError('get not supported yet')
    if idx_aval.size != size:
      raise NotImplementedError('get not supported yet')
    bd = next(block_shape_iter)
    block_shape.append(_block_size(bd))
  assert next(block_shape_iter, None) is None

  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    idx_iter = iter(idx)
    indices = tuple(
        0
        if (bd is None or isinstance(bd, pallas_core.Squeezed))
        else next(idx_iter)
        for bd in range(len(block_shape))
    )
    assert next(idx_iter, None) is None
    return indices

  block_spec = pallas_core.BlockSpec(block_shape, new_index_map)
  return [block_spec] + [pallas_core.no_block_spec] * (len(ctx.avals_in) - 1)


@register_eval_rule(state_primitives.get_p)
def _get_eval_rule(ctx: KernelEvalContext, ref, *idx, tree):
  indexers = tree_util.tree_unflatten(tree, idx)
  ref_aval = ctx.avals_in[0]
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[1:])
  ref_block_spec = ctx.in_block_specs[0]
  assert hasattr(ref_aval, 'shape')
  if len(indexers) > 1:
    raise NotImplementedError('get not supported yet')
  indexer = indexers[0]
  indexer_aval = indexers_avals[0]
  block_indexer = []

  def _slice(i, b):
    match b:
      case int():
        return indexing.ds(i * b, b)
      case pallas_core.Blocked(bs):
        return indexing.ds(i * bs, bs)
      case pallas_core.Squeezed() | None:
        return i
      case _:
        raise NotImplementedError('get not supported yet')

  if ref_block_spec is pallas_core.no_block_spec:
    # Short-circuit if the ref is not blocked.
    return state_primitives.get_p.bind(ref, *idx, tree=tree)
  block_idx_iter = iter(ctx.get_out_block_indices()[0])
  for idx_aval, size, idx, bd in zip(
      indexer_aval.indices,
      ref_aval.shape,
      indexer.indices,
      ref_block_spec.block_shape,
      strict=True,
  ):
    if not isinstance(idx_aval, indexing.Slice):
      assert hasattr(idx_aval, 'shape') and not idx_aval.shape, idx_aval
      assert bd is None or isinstance(bd, pallas_core.Squeezed)
      block_indexer.append(idx)
      continue
    if not isinstance(idx_aval.start, int):
      raise NotImplementedError('get not supported yet')
    if not isinstance(idx_aval.size, int):
      raise NotImplementedError('get not supported yet')
    if idx_aval.stride != 1:
      raise NotImplementedError('get not supported yet')
    if idx_aval.start != 0:
      raise NotImplementedError('get not supported yet')
    if idx_aval.size != size:
      raise NotImplementedError('get not supported yet')
    bidx = next(block_idx_iter)
    block_indexer.append(_slice(bidx, bd))
  assert next(block_idx_iter, None) is None
  return ref.get(idx=tuple(block_indexer))


@register_eval_rule(lax.concatenate_p)
def _concatenate_eval_rule(ctx: KernelEvalContext, *args, dimension):
  # We now handle the case where each of the concatenated array dimensions
  # divides the block size.
  block_spec = ctx.out_block_specs[0]
  block_shape = block_spec.block_shape
  is_element_block = [isinstance(bd, pallas_core.Element) for bd in block_shape]
  if any(is_element_block):
    raise NotImplementedError(
        'Concatenation with Element indexing is not yet supported.'
    )
  block_dim = block_shape[dimension]
  if block_dim is None:
    block_dim = 1

  if block_dim == sum(aval.shape[dimension] for aval in ctx.avals_in):  # pytype: disable=attribute-error
    # Handle special case if the block contains all of the concatenated
    # array.
    return jax.lax.concatenate(args, dimension=dimension)

  num_blocks = []
  for aval in ctx.avals_in:
    assert isinstance(aval, core.ShapedArray)
    if aval.shape[dimension] % block_dim != 0:
      raise ValueError(
          f'Shape along concat dimension {dimension} must be divisible by the'
          f' block shape {block_shape[dimension]} for all children. Got shape'
          f' {aval.shape}.'
      )
    num_blocks.append(aval.shape[dimension] // block_dim)
  ends = np.cumsum(num_blocks).astype(np.int32)
  starts = np.concatenate(([0], ends[:-1])).astype(np.int32)

  block_indices = ctx.get_out_block_indices()[0]
  block_idx = block_indices[dimension]
  valid_index = 0
  for i in range(len(ctx.avals_in)):
    start, end = starts[i], ends[i]
    is_valid = (start <= block_idx) & (block_idx < end)
    valid_index = jax.lax.select(is_valid, i, valid_index)
  out_dtype = args[0].dtype
  args = [a.astype(jnp.float32) if a.dtype == jnp.bfloat16 else a for a in args]
  valid_block = jax.lax.select_n(valid_index, *args)
  return valid_block.astype(out_dtype)


@register_pull_block_spec_rule(lax.concatenate_p)
def _concatenate_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dimension: int,
):
  block_shape = block_spec.block_shape
  is_element_block = [isinstance(bd, pallas_core.Element) for bd in block_shape]
  if any(is_element_block):
    raise NotImplementedError(
        'Concatenation with Element indexing is not yet supported.'
    )
  num_blocks = []
  block_dim = block_shape[dimension]
  if block_dim is None or isinstance(block_dim, pallas_core.Squeezed):
    block_dim = 1
  if block_dim == sum(aval.shape[dimension] for aval in ctx.avals_in):  # pytype: disable=attribute-error
    # Handle special case if the block contains all of the concatenated
    # array.
    new_shapes = [
        util.tuple_update(  # pytype: disable=wrong-arg-types
            block_spec.block_shape, dimension, aval.shape[dimension]  # pytype: disable=attribute-error
        )
        for aval in ctx.avals_in
    ]
    new_block_specs = [
        block_spec.replace(block_shape=shape) for shape in new_shapes
    ]
    return new_block_specs

  # We now handle the case where each of the concatenated array dimensions
  # divides the block size.
  for aval in ctx.avals_in:
    assert isinstance(aval, core.ShapedArray)
    if aval.shape[dimension] % block_dim != 0:
      raise ValueError(
          f'Shape along concat dimension {dimension} must be divisible by the'
          f' block shape {block_shape[dimension]} for all children. Got shape'
          f' {aval.shape}.'
      )
    num_blocks.append(aval.shape[dimension] // block_dim)
  ends = np.cumsum(num_blocks).astype(np.int32)
  starts = np.concatenate(([0], ends[:-1])).astype(np.int32)

  def make_block_spec(child_index: int):
    def new_index_map(*args):
      idx = block_spec.index_map(*args)
      block_idx = idx[dimension]
      is_valid = (starts[child_index] <= block_idx) & (
          block_idx < ends[child_index]
      )
      padding_index = jnp.where(
          block_idx < starts[child_index], 0, num_blocks[child_index] - 1
      )
      block_idx = jnp.where(
          is_valid, block_idx - starts[child_index], padding_index
      )
      return util.tuple_update(idx, dimension, block_idx)

    return pallas_core.BlockSpec(block_spec.block_shape, new_index_map)

  return [make_block_spec(i) for i in range(len(ctx.avals_in))]


@register_usage_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_usage_rule(ctx, used_out: set[Usage], **params):
  del params
  if used_out == {Usage.SCALAR_PREFETCH}:
    raise NotImplementedError('scalar prefetch not supported yet')
  elif used_out == {Usage.REGULAR}:
    return [
        {Usage.SCALAR_PREFETCH}
        if not ctx.avals_in[0].shape
        else {Usage.REGULAR}
    ]
  else:
    return [set()]


@register_eval_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_eval_rule(
    eval_ctx: KernelEvalContext, x, broadcast_dimensions, shape, **params
):
  del params  # Unused.
  in_shape = eval_ctx.avals_in[0].shape  # pytype: disable=attribute-error
  if in_shape == shape:
    # Dummy broadcast
    return x
  shape = tuple(map(_block_size, eval_ctx.out_block_specs[0].block_shape))
  dims = tuple(
      d - sum(s is None for s in shape[:d])
      for d in broadcast_dimensions
      if shape[d] is not None
  )
  shape = tuple(s for s in shape if s is not None)
  return jax.lax.broadcast_in_dim(x, broadcast_dimensions=dims, shape=shape)


@register_pull_block_spec_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
    sharding: jax.sharding.Sharding,
):
  del shape, sharding

  shape = ctx.avals_in[0].shape  # pytype: disable=attribute-error
  if not shape:
    return [pallas_core.no_block_spec]

  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    return tuple(
        0 if (d == 1) else idx[i]
        for i, d in zip(broadcast_dimensions, shape, strict=True)
    )

  new_block_shape = tuple(
      b if ((b := block_spec.block_shape[i]) is None) or (d != 1) else 1
      for i, d in zip(broadcast_dimensions, shape, strict=True)
  )
  return [pallas_core.BlockSpec(new_block_shape, new_index_map)]


@register_eval_rule(lax.transpose_p)
def _transpose_eval_rule(
    eval_ctx: KernelEvalContext, x, permutation: tuple[int, ...]
):
  block_spec = eval_ctx.out_block_specs[0]
  block_shape = block_spec.block_shape
  block_shape_no_nones = tuple(
      bs
      for bs in block_shape
      if not (bs is None or isinstance(bs, pallas_core.Squeezed))
  )
  block_dims_iter = iter(range(len(block_shape_no_nones)))
  expanded_block_dims = [
      None
      if (bs is None or isinstance(bs, pallas_core.Squeezed))
      else next(block_dims_iter)
      for bs in block_shape
  ]
  assert next(block_dims_iter, None) is None
  permuted_block_dims = [expanded_block_dims[p] for p in permutation]
  new_permutation = [p for p in permuted_block_dims if p is not None]
  return jax.lax.transpose(x, permutation=new_permutation)


@register_pull_block_spec_rule(lax.transpose_p)
def _transpose_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    permutation: tuple[int, ...],
):

  block_shape = block_spec.block_shape
  new_shape = tuple(block_shape[i] for i in permutation)
  aval_in = ctx.avals_in[0]
  assert isinstance(aval_in, core.ShapedArray)
  assert len(block_shape) == len(aval_in.shape)
  if set(permutation[-2:]) != {permutation[-1], permutation[-2]}:
    raise NotImplementedError(
        'Cannot permute last two dimensions with leading dimensions.'
    )

  def new_index_map(*args):
    original_idxs = block_spec.index_map(*args)
    return tuple(original_idxs[i] for i in permutation)

  return [pallas_core.BlockSpec(new_shape, new_index_map)]


@register_eval_rule(lax.convert_element_type_p)
def _convert_element_type_eval_rule(
    eval_ctx: KernelEvalContext, x, new_dtype, **params
):
  return jax.lax.convert_element_type(x, new_dtype)


@register_pull_block_spec_rule(lax.convert_element_type_p)
def _convert_element_type_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    new_dtype: jnp.dtype,
    weak_type: bool,
    sharding: jax.sharding.Sharding,
):
  del ctx, new_dtype, weak_type, sharding
  return [block_spec]


@register_eval_rule(lax.bitcast_convert_type_p)
def _bitcast_convert_type_eval_rule(eval_ctx: KernelEvalContext, x, new_dtype):
  return jax.lax.bitcast_convert_type(x, new_dtype)


@register_pull_block_spec_rule(lax.bitcast_convert_type_p)
def _bitcast_convert_type_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    new_dtype: jnp.dtype,
):
  old_dtype = ctx.avals_in[0].dtype  # pytype: disable=attribute-error
  if old_dtype.itemsize != new_dtype.itemsize:
    raise NotImplementedError(
        'bitcast_convert_type with different bitwidths not supported yet:'
        f' {old_dtype=}, {new_dtype=}'
    )
  return [block_spec]


@register_eval_rule(prng.random_bits_p)
def _random_bits_eval_rule(eval_ctx: KernelEvalContext, key, bit_width, shape):
  del shape
  block_spec = eval_ctx.out_block_specs[0]
  indices = eval_ctx.get_out_block_indices()[0]
  block_shape = block_spec.block_shape
  # This is the important part here: we fold in block indices into the key so
  # each block gets different random numbers.
  for idx in indices:
    key = jax.random.fold_in(key, idx)
  return prng.random_bits(key, bit_width=bit_width, shape=block_shape)


@register_pull_block_spec_rule(prng.random_bits_p)
def _random_bits_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    **_,
):
  del ctx, block_spec
  key_block_spec = pallas_core.BlockSpec(
      block_shape=None, memory_space=pallas_core.MemorySpace.KEY
  )
  return [key_block_spec]


@register_eval_rule(prng.random_wrap_p)
def _random_wrap_eval_rule(eval_ctx: KernelEvalContext, arr, *, impl):
  del eval_ctx
  return jax.random.wrap_key_data(arr, impl=impl)


@register_pull_block_spec_rule(prng.random_wrap_p)
def _random_wrap_pull_rule(
    ctx: PullRuleContext, block_spec: pallas_core.BlockSpec, *, impl
):
  del ctx, block_spec, impl
  return [pallas_core.BlockSpec(block_shape=None)]


@register_eval_rule(lax.iota_p)
def _iota_eval_rule(
    eval_ctx: KernelEvalContext, *, dimension, shape, dtype, sharding
):
  del sharding
  block_spec = eval_ctx.out_block_specs[0]
  block_idx = eval_ctx.get_out_block_indices()[0]
  assert len(block_idx) == len(shape)
  iota_shape = tuple(
      _block_size(s) for s in block_spec.block_shape if s is not None
  )
  dim_ = dimension - sum(
      _block_size(s) is None for s in block_spec.block_shape[:dimension]
  )
  local_iota = jax.lax.broadcasted_iota(dtype, iota_shape, dim_)
  return local_iota + block_idx[dimension] * _block_size(
      block_spec.block_shape[dimension]
  )


@register_pull_block_spec_rule(lax.iota_p)
def _iota_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dtype: jnp.dtype,
    dimension: int,
    shape: tuple[int, ...],
    sharding: jax.sharding.Sharding,
):
  del ctx, sharding, dtype, shape
  if block_spec.block_shape[dimension] is None:
    raise ValueError(
        f'Cannot pull iota along dimension {dimension} with None block size.'
    )
  return []


def _pattern_match_lanes_to_sublanes_reshape(
    aval_in: core.ShapedArray,
    aval_out: core.ShapedArray,
) -> bool:
  # Pattern matches a reshape of the form (..., n * l) -> (..., n, l)
  # where l is a multiple of 128.

  *leading_out, last_dim_in = aval_in.shape
  *leading_in, second_to_last_dim_out, last_dim = aval_out.shape
  if leading_in != leading_out:
    return False
  if second_to_last_dim_out * last_dim != last_dim_in:
    return False
  if last_dim % 128 != 0:
    return False
  return True


@register_pull_block_spec_rule(lax.reshape_p)
def _reshape_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dimensions: tuple[int, ...] | None,
    new_sizes: tuple[int, ...],
    sharding: jax.sharding.Sharding,
):
  del sharding, new_sizes
  if dimensions is not None:
    raise NotImplementedError('reshape with None dimensions not supported yet')
  aval_in = ctx.avals_in[0]
  assert isinstance(aval_in, core.ShapedArray)
  aval_out = ctx.avals_out[0]
  assert isinstance(aval_out, core.ShapedArray)

  block_shape = tuple(map(_block_size, block_spec.block_shape))
  shape_in = aval_in.shape
  shape_out = aval_out.shape
  assert np.prod(shape_in) == np.prod(shape_out)

  # Handle merged dims; i.e. (..., m, n, ...) -> (..., m * n, ...).
  i = 0
  j = 0
  merged_dims = []

  while i < len(shape_in) and j < len(shape_out):
    merged = []
    while not merged or np.prod(merged) < shape_out[j]:
      merged.append(shape_in[i])
      i += 1

    if np.prod(merged) > shape_out[j]:
      break  # Dimension has been split (or something more complex).

    merged_dims.append(merged)
    j += 1

  if (i == len(shape_in)) and np.prod(shape_out[j:]) == 1:
    new_block_shape = []
    new_grids = []

    for d, bd, merged in zip(shape_out, block_shape, merged_dims):
      if not isinstance(bd, (int, pallas_core.Blocked)):
        raise NotImplementedError('reshape merge must use `Blocked` block size')

      num_blocks = pallas_utils.cdiv(d, bd)
      new_block_dims = []
      for md in reversed(merged):
        if bd % md == 0:
          new_block_dims.append(md)
          bd //= md
        elif md % bd == 0:
          new_block_dims.append(bd)
          bd = 1
        else:
          raise NotImplementedError('unsupported reshape merge')

      new_block_dims.reverse()
      new_block_shape.extend(new_block_dims)
      new_grid = [np.int32(md // bd) for md, bd in zip(merged, new_block_dims)]
      new_grids.append(tuple(new_grid))

      if np.prod(new_grid) != num_blocks:
        raise NotImplementedError('reshape merge must maintain grid size')

    def new_index_map(*args):
      # NOTE: The `zip` will drop indices for any trailing `1` dims.
      idxs = (
          jnp.unravel_index(idx, new_grid) if len(new_grid) > 1 else (idx,)
          for idx, new_grid in zip(block_spec.index_map(*args), new_grids)
      )
      return sum(idxs, ())

    return [pallas_core.BlockSpec(tuple(new_block_shape), new_index_map)]

  # Handle the case where we reshape from (..., n * l) -> (..., n, l)
  if _pattern_match_lanes_to_sublanes_reshape(aval_in, aval_out):
    if not isinstance(block_shape[-1], (int, pallas_core.Blocked)):
      raise NotImplementedError(
          f'reshape must use Blocked block size on lanes: {block_shape}'
      )
    if not isinstance(block_shape[-2], (int, pallas_core.Blocked)):
      raise NotImplementedError(
          f'reshape must use Blocked block size on sublanes: {block_shape}'
      )
    last_dim = aval_out.shape[-1]
    block_sublane_dim, block_lane_dim = (
        _block_size(block_shape[-2]),
        _block_size(block_shape[-1]),
    )
    total_block_size = block_sublane_dim * block_lane_dim
    if total_block_size % 128 != 0:
      raise NotImplementedError(
          'reshape with non-128 aligned block size on lanes not supported yet'
      )
    if block_lane_dim != last_dim:
      raise NotImplementedError(
          'reshape with non-matching block size on lanes not supported yet:'
          f' {block_shape}'
      )
    new_block_shape = block_shape[:-2] + (total_block_size,)
    def new_index_map(*args):  # pylint: disable=function-redefined
      *idx, second_to_last, last = block_spec.index_map(*args)
      # last should always be 0
      if not isinstance(last, int) and last != 0:
        raise NotImplementedError(
            'Must select entire block on last dimension for reshape'
        )
      return *idx, second_to_last
    return [pallas_core.BlockSpec(new_block_shape, new_index_map)]

  raise NotImplementedError(f'reshape not supported yet: {aval_in}, {aval_out}')


@register_eval_rule(lax.reshape_p)
def _reshape_eval_rule(
    eval_ctx: KernelEvalContext, x, *, dimensions, new_sizes, sharding
):
  del sharding, dimensions, new_sizes
  out_shape_nones = tuple(
      _block_size(s) for s in eval_ctx.out_block_specs[0].block_shape
  )
  out_shape = tuple(s for s in out_shape_nones if s is not None)
  # Because we have restricted the pull block spec rule, we can just apply a
  # basic reshape here.
  x = x.reshape(out_shape)
  return x


@register_pull_block_spec_rule(lax.reduce_sum_p)
def _reduce_sum_pull_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    axes: tuple[int, ...],
):
  aval_in = ctx.avals_in[0]
  assert isinstance(aval_in, core.ShapedArray)
  new_block_shape = []
  block_shape = iter(block_spec.block_shape)
  for i, d in enumerate(aval_in.shape):
    if i in axes:
      new_block_shape.append(pallas_core.Blocked(d))
    else:
      new_block_shape.append(next(block_shape))
  assert next(block_shape, None) is None
  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    new_idx = []
    idx_iter = iter(idx)
    for i in range(len(aval_in.shape)):
      if i in axes:
        new_idx.append(0)
      else:
        new_idx.append(next(idx_iter))
    assert next(idx_iter, None) is None
    return tuple(new_idx)
  new_block_spec = block_spec.replace(block_shape=tuple(new_block_shape),
                                      index_map=new_index_map)
  return [new_block_spec]


@register_eval_rule(lax.reduce_sum_p)
def _reduce_sum_eval_rule(
    ctx: KernelEvalContext,
    x,
    *,
    axes: tuple[int, ...],
):
  aval_in = ctx.avals_in[0]
  assert isinstance(aval_in, core.ShapedArray)
  block_shape = tuple(ctx.in_block_specs[0].block_shape)
  for i in axes:
    if _block_size(block_shape[i]) != aval_in.shape[i]:
      raise NotImplementedError(
          f'reduce_sum on partial blocks not supported: {aval_in=},'
          f' {block_shape=}'
      )
  return jax.lax.reduce_sum(x, axes=axes)


# Higher order primitives


@register_usage_rule(pjit.jit_p)
def _jit_usage_rule(
    ctx, used_out: list[set[Usage]], *, jaxpr: core.ClosedJaxpr, **_
):
  read_usage_env = compute_usage(jaxpr.jaxpr, used_out)
  in_usages = util.safe_map(read_usage_env, jaxpr.jaxpr.invars)
  return in_usages


@register_eval_rule(pjit.jit_p)
def _jit_eval_rule(ctx: KernelEvalContext, *args, jaxpr, **kwargs):
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  if consts:
    raise NotImplementedError('pjit with consts not supported yet')
  out_tree = tree_util.tree_structure(tuple(jaxpr.outvars))
  in_tree = tree_util.tree_structure((tuple(jaxpr.invars), {}))

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  _, env, _ = _pull_block_spec(
      jaxpr,
      ctx.out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      read_usage_env=read_usage_env,
      grid=ctx.grid,
  )
  kernel_fn = make_kernel_function(
      jaxpr,
      (),
      in_tree,
      out_tree,
      read_usage_env,
      ctx.in_block_specs,
      env,
      ctx.scalar_prefetch_handler,
      ctx.grid,
  )
  return kernel_fn(ctx.get_program_ids(), ctx.scalar_prefetch, *args)


@register_pull_block_spec_rule(pjit.jit_p)
def _jit_pull_block_spec_rule(
    ctx: PullRuleContext, out_block_specs, *, jaxpr, **kwargs
):
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  if consts:
    raise NotImplementedError('pjit with consts not supported yet')

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  in_block_specs, _, _ = _pull_block_spec(
      jaxpr,
      out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      read_usage_env=read_usage_env,
      grid=ctx.grid,
  )
  return in_block_specs


@register_usage_rule(custom_derivatives.custom_jvp_call_p)
def _custom_jvp_call_usage_rule(
    ctx, used_out: list[set[Usage]], *, call_jaxpr: core.ClosedJaxpr, **_
):
  del ctx
  read_usage_env = compute_usage(call_jaxpr.jaxpr, used_out)
  in_usages = util.safe_map(read_usage_env, call_jaxpr.jaxpr.invars)
  return in_usages


@register_eval_rule(custom_derivatives.custom_jvp_call_p)
def _custom_jvp_call_eval_rule(
    ctx: KernelEvalContext, *args, call_jaxpr: core.ClosedJaxpr, **kwargs
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError('custom_jvp_call with consts not supported yet')
  out_tree = tree_util.tree_structure(tuple(jaxpr.outvars))
  in_tree = tree_util.tree_structure((tuple(jaxpr.invars), {}))

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  _, env, _ = _pull_block_spec(
      jaxpr,
      ctx.out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
      read_usage_env=read_usage_env,
  )
  kernel_fn = make_kernel_function(
      jaxpr,
      (),
      in_tree,
      out_tree,
      read_usage_env,
      ctx.in_block_specs,
      env,
      ctx.scalar_prefetch_handler,
      ctx.grid,
  )
  return kernel_fn(ctx.get_program_ids(), ctx.scalar_prefetch, *args)


@register_pull_block_spec_rule(custom_derivatives.custom_jvp_call_p)
def _custom_jvp_call_pull_block_spec_rule(
    ctx: PullRuleContext, out_block_specs, *, call_jaxpr, **kwargs
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError('custom_jvp_call with consts not supported yet')

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  in_block_specs, _, _ = _pull_block_spec(
      jaxpr,
      out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
      read_usage_env=read_usage_env,
  )
  return in_block_specs


@register_usage_rule(custom_derivatives.custom_vjp_call_p)
def _custom_vjp_call_usage_rule(
    ctx, used_out: list[set[Usage]], *, call_jaxpr: core.ClosedJaxpr, **_
):
  del ctx
  read_usage_env = compute_usage(call_jaxpr.jaxpr, used_out)
  in_usages = util.safe_map(read_usage_env, call_jaxpr.jaxpr.invars)
  return in_usages


@register_eval_rule(custom_derivatives.custom_vjp_call_p)
def _custom_vjp_call_eval_rule(
    ctx: KernelEvalContext, *args, call_jaxpr: core.ClosedJaxpr, **kwargs
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError('custom_vjp_call with consts not supported yet')
  out_tree = tree_util.tree_structure(tuple(jaxpr.outvars))
  in_tree = tree_util.tree_structure((tuple(jaxpr.invars), {}))

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  _, env, _ = _pull_block_spec(
      jaxpr,
      ctx.out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
      read_usage_env=read_usage_env,
  )
  kernel_fn = make_kernel_function(
      jaxpr,
      (),
      in_tree,
      out_tree,
      read_usage_env,
      ctx.in_block_specs,
      env,
      ctx.scalar_prefetch_handler,
      ctx.grid,
  )
  return kernel_fn(ctx.get_program_ids(), ctx.scalar_prefetch, *args)


@register_pull_block_spec_rule(custom_derivatives.custom_vjp_call_p)
def _custom_vjp_call_pull_block_spec_rule(
    ctx: PullRuleContext, out_block_specs, *, call_jaxpr, **kwargs
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError('custom_vjp_call with consts not supported yet')

  def read_usage_env(_: core.Var):
    return {Usage.REGULAR}

  in_block_specs, _, _ = _pull_block_spec(
      jaxpr,
      out_block_specs,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
      read_usage_env=read_usage_env,
  )
  return in_block_specs


def push_block_spec(
    f: Callable,
    *in_spec_args,
    **in_spec_kwargs,
):
  def wrapper(*args, **kwargs):
    flat_block_specs, in_tree_ = tree_util.tree_flatten(
        (in_spec_args, in_spec_kwargs)
    )
    jaxpr, _, in_tree, out_tree = fuser_utils.make_jaxpr(f, *args, **kwargs)
    if in_tree != in_tree_:
      raise ValueError(f'Expected {in_tree} PyTree, got {in_tree_}')
    out_bs = _push_block_spec_jaxpr(jaxpr, *flat_block_specs)
    return tree_util.tree_unflatten(out_tree, out_bs)

  return wrapper


def _push_block_spec_jaxpr(
    jaxpr: core.Jaxpr,
    *flat_block_specs,
) -> tuple[pallas_core.BlockSpec, ...]:
  num_inputs = len(jaxpr.invars)
  if len(flat_block_specs) != num_inputs:
    raise ValueError(
        f'Expected {num_inputs} block specs, got {len(flat_block_specs)}'
    )

  env: dict[core.Var, pallas_core.BlockSpec | pallas_core.NoBlockSpec] = {}

  for invar, bs in zip(jaxpr.invars, flat_block_specs, strict=True):
    env[invar] = bs
  for constvar in jaxpr.constvars:
    env[constvar] = pallas_core.no_block_spec

  def _read_block_spec(
      atom: core.Atom,
  ) -> pallas_core.BlockSpec | pallas_core.NoBlockSpec:
    if isinstance(atom, core.Literal):
      return pallas_core.no_block_spec
    return env[atom]

  def _write_block_spec(
      atom: core.Atom,
      block_spec: pallas_core.BlockSpec | pallas_core.NoBlockSpec,
  ):
    if isinstance(atom, core.Literal):
      return
    env[atom] = block_spec

  for eqn in jaxpr.eqns:
    in_block_specs = tuple(util.safe_map(_read_block_spec, eqn.invars))
    if all(bs is pallas_core.no_block_spec for bs in in_block_specs):
      for outvar in eqn.outvars:
        _write_block_spec(outvar, pallas_core.no_block_spec)
      continue
    rule = push_block_spec_rules.get(eqn.primitive, None)
    if not rule:
      raise NotImplementedError(eqn.primitive)
    ctx = PushRuleContext(
        avals_in=tuple(v.aval for v in eqn.invars),
        avals_out=tuple(v.aval for v in eqn.outvars),
    )
    if eqn.primitive.multiple_results:
      out_block_specs = rule(ctx, *in_block_specs, **eqn.params)
    else:
      out_block_specs = [rule(ctx, *in_block_specs, **eqn.params)]

    util.safe_map(_write_block_spec, eqn.outvars, out_block_specs)
  out_block_specs = tuple(util.safe_map(_read_block_spec, jaxpr.outvars))
  valid_block_spec = [
      bs for bs in flat_block_specs if bs is not pallas_core.no_block_spec
  ][0]
  out_block_specs = tuple(
      valid_block_spec if obs is pallas_core.no_block_spec else obs
      for obs in out_block_specs
  )
  if any(bs is pallas_core.no_block_spec for bs in out_block_specs):
    raise ValueError('No block spec found for output')
  return out_block_specs  # pytype: disable=bad-return-type


push_block_spec_rules: dict[core.Primitive, PushBlockSpecRuleFn] = {}


class PushBlockSpecRuleFn(Protocol):

  def __call__(
      self,
      ctx: PushRuleContext,
      block_spec: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
      **params: Any,
  ) -> pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...]:
    ...


def register_push_block_spec_rule(
    prim: core.Primitive,
) -> Callable[[PushBlockSpecRuleFn], PushBlockSpecRuleFn]:

  def wrapper(
      f: PushBlockSpecRuleFn,
  ) -> PushBlockSpecRuleFn:
    push_block_spec_rules[prim] = f
    return f

  return wrapper


def _binop_push_rule(
    prim: core.Primitive,
    ctx: PullRuleContext,
    left_block_spec: pallas_core.BlockSpec,
    right_block_spec: pallas_core.BlockSpec,
    **params: Any,
) -> Sequence[pallas_core.BlockSpec]:
  del prim, params
  left_aval, right_aval = ctx.avals_in
  assert isinstance(left_aval, core.ShapedArray)
  assert isinstance(right_aval, core.ShapedArray)
  if left_block_spec is pallas_core.no_block_spec:
    return right_block_spec
  if right_block_spec is pallas_core.no_block_spec:
    return left_block_spec
  if right_block_spec != left_block_spec:
    raise ValueError('Invalid block spec')
  return left_block_spec


register_binop_push_rule = lambda prim: register_push_block_spec_rule(prim)(
    functools.partial(_binop_push_rule, prim)
)
register_binop_push_rule(lax.mul_p)
register_binop_push_rule(lax.add_p)
register_binop_push_rule(lax.sub_p)
register_binop_push_rule(lax.div_p)
register_binop_push_rule(lax.max_p)
register_binop_push_rule(lax.lt_p)
register_binop_push_rule(lax.eq_p)
register_binop_push_rule(lax.gt_p)
register_binop_push_rule(lax.and_p)
register_binop_push_rule(lax.pow_p)
register_binop_push_rule(ad_util.add_any_p)


def _eltwise_push_rule(
    prim: core.Primitive,
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    **params,
) -> pallas_core.BlockSpec:
  del prim, ctx, params
  return block_spec


@register_push_block_spec_rule(lax.transpose_p)
def _transpose_push_rule(
    ctx: PushRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    permutation: tuple[int, ...],
) -> pallas_core.BlockSpec:
  del ctx
  block_shape = block_spec.block_shape
  new_shape = [block_shape[i] for i in permutation]
  if set(permutation[-2:]) != {permutation[-1], permutation[-2]}:
    raise NotImplementedError(
        'Cannot permute last two dimensions with leading dimensions.'
    )

  def new_index_map(*args):
    original_idxs = block_spec.index_map(*args)
    return tuple(original_idxs[i] for i in permutation)

  return pallas_core.BlockSpec(new_shape, new_index_map)


@register_push_block_spec_rule(lax.convert_element_type_p)
def _convert_element_type_push_rule(
    ctx: PushRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    new_dtype: jnp.dtype,
    weak_type: bool,
    sharding: jax.sharding.Sharding,
):
  del ctx, new_dtype, weak_type, sharding
  return block_spec


@register_push_block_spec_rule(lax.select_n_p)
def _select_n_push_rule(
    ctx: PushRuleContext,
    *args: pallas_core.BlockSpec,
):
  del ctx
  block_specs = [b for b in args if b is not pallas_core.no_block_spec]
  assert len(block_specs) > 0
  block_spec = block_specs[0]
  if len(block_specs) > 1:
    if any(b is not block_spec for b in block_specs):
      raise NotImplementedError(
          'select_n with multiple differing inputs not supported yet'
      )
  return block_spec


@register_push_block_spec_rule(custom_derivatives.custom_jvp_call_p)
def _custom_jvp_call_push_rule(
    ctx, *block_specs, call_jaxpr: core.ClosedJaxpr, **_
):
  assert not call_jaxpr.consts
  return _push_block_spec_jaxpr(call_jaxpr.jaxpr, *block_specs)


@register_push_block_spec_rule(custom_derivatives.custom_vjp_call_p)
def _custom_vjp_call_push_rule(
    ctx,
    *block_specs,
    call_jaxpr: core.ClosedJaxpr,
    num_consts,
    fwd_jaxpr_thunk,
    bwd,
    out_trees,
    symbolic_zeros,
):
  del ctx, num_consts, fwd_jaxpr_thunk, bwd, out_trees, symbolic_zeros
  return _push_block_spec_jaxpr(call_jaxpr.jaxpr, *block_specs)


@register_push_block_spec_rule(pjit.jit_p)
def _pjit_push_rule(ctx, *block_specs, jaxpr: core.ClosedJaxpr, **_):
  assert not jaxpr.consts
  return _push_block_spec_jaxpr(jaxpr.jaxpr, *block_specs)


def register_eltwise_rule(prim: core.Primitive):
  register_pull_block_spec_rule(prim)(
      functools.partial(_eltwise_pull_rule, prim)
  )
  register_usage_rule(prim)(functools.partial(_eltwise_usage_rule, prim))
  register_eval_rule(prim)(functools.partial(_eltwise_eval_rule, prim))
  register_push_block_spec_rule(prim)(
      functools.partial(_eltwise_push_rule, prim)
  )


register_eltwise_rule(lax.exp_p)
register_eltwise_rule(lax.tanh_p)
register_eltwise_rule(lax.sin_p)
register_eltwise_rule(lax.cos_p)
register_eltwise_rule(lax.sqrt_p)
register_eltwise_rule(lax.rsqrt_p)
register_eltwise_rule(lax.log_p)
register_eltwise_rule(lax.integer_pow_p)

@register_push_block_spec_rule(lax.reshape_p)
def _reshape_push_rule(
    ctx: PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dimensions: tuple[int, ...] | None,
    new_sizes: tuple[int, ...],
    sharding: jax.sharding.Sharding,
):
  del sharding, new_sizes
  if dimensions is not None:
    raise NotImplementedError('reshape with None dimensions not supported yet')
  aval_in = ctx.avals_in[0]
  assert isinstance(aval_in, core.ShapedArray)
  aval_out = ctx.avals_out[0]
  assert isinstance(aval_out, core.ShapedArray)
  if _pattern_match_lanes_to_sublanes_reshape(aval_in, aval_out):
    block_shape = tuple(block_spec.block_shape)
    if not isinstance(block_shape[-1], (int, pallas_core.Blocked)):
      raise NotImplementedError(
          f'reshape must use Blocked block size on lanes: {block_shape}'
      )
    last_dim = aval_out.shape[-1]
    last_block_dim = _block_size(block_shape[-1])
    if last_block_dim % 128 != 0:
      raise NotImplementedError(
          'reshape with non-128 aligned block size on lanes not supported yet'
      )
    if last_block_dim % last_dim != 0:
      raise NotImplementedError(
          'reshape with non-divisible block size on lanes not supported yet'
      )
    num_last_dim_blocks = last_block_dim // last_dim
    new_block_shape = block_shape[:1] + (num_last_dim_blocks, last_dim)

    def new_index_map(*args):
      *idx, last = block_spec.index_map(*args)
      return *idx, last, 0

    return pallas_core.BlockSpec(new_block_shape, new_index_map)
  raise NotImplementedError(f'reshape not supported yet: {aval_in}, {aval_out}')


@register_push_block_spec_rule(lax.reduce_sum_p)
def _reduce_sum_push_rule(
    ctx: PushRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    axes: tuple[int, ...],
):
  del ctx
  if axes:
   raise NotImplementedError('reduce_sum with no axes not supported yet')
  return block_spec
