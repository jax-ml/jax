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

import contextlib
import dataclasses
import enum
import functools
import threading
from typing import Any, Callable, Protocol, Sequence

import jax
from jax import lax
from jax._src import ad_util
from jax._src import core
from jax._src import custom_derivatives
from jax._src import pjit
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import fuser_utils
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


def _default_eval_fn(eqn, eval_ctx, *args):
  del eval_ctx
  out = eqn.primitive.bind(*args, **eqn.params)
  if eqn.primitive.multiple_results:
    return out
  return [out]


def _wrap_eval_fn(primitive, eval_fn):
  def wrapped(*args):
    if primitive.multiple_results:
      return eval_fn(*args)
    return [eval_fn(*args)]

  return wrapped


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

_sp_env = threading.local()
_sp_env.scalar_prefetch = None


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
  if block_spec is pallas_core.no_block_spec:
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
    # TODO(sharadmv): handle these consts better, they should correspond to
    # scalar prefetch.
    del consts, out_tree_
    jaxpr_out_usages = [{Usage.REGULAR}] * len(jaxpr.outvars)
    block_specs_ = jax.tree.map(
        _unwrap_block_spec_scalar_prefetch, out_block_specs
    )
    flat_block_specs, out_tree = jax.tree.flatten(block_specs_)
    in_block_specs, env, read_usage_env = _pull_block_spec(
        jaxpr,
        tuple(flat_block_specs),
        jaxpr_out_usages,
        scalar_prefetch_handler=scalar_prefetch_handler,
        grid=grid,
    )
    kernel_fn = make_kernel_function(
        jaxpr,
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
    out_usages,
    *,
    scalar_prefetch_handler: Any | None = None,
    grid: tuple[int | jax.Array, ...],
) -> tuple[
    tuple[pallas_core.BlockSpec | pallas_core.NoBlockSpec, ...],
    tuple[dict[core.Var, pallas_core.BlockSpec], dict[int, Any]],
    Any,
]:
  read_usage_env = compute_usage(jaxpr, out_usages)
  jaxpr_invar_usages = util.safe_map(read_usage_env, jaxpr.invars)
  env: dict[core.Var, pallas_core.BlockSpec] = {}
  scalar_prefetch_fn_env = {}

  for outvar, bs in zip(jaxpr.outvars, out_block_specs, strict=True):
    assert isinstance(outvar, core.Var)
    env[outvar] = bs

  def _read_block_spec(atom: core.Atom) -> pallas_core.BlockSpec | Any:
    if isinstance(atom, core.Literal):
      return pallas_core.no_block_spec
    return env[atom]

  def _write_block_spec(atom: core.Atom, block_spec: pallas_core.BlockSpec):
    if isinstance(atom, core.Literal):
      return
    env[atom] = block_spec

  for i, eqn in reversed(list(enumerate(jaxpr.eqns))):
    eqn_out_block_specs = tuple(util.safe_map(_read_block_spec, eqn.outvars))
    rule = pull_block_spec_rules.get(eqn.primitive, None)
    if not rule:
      raise NotImplementedError(eqn.primitive)
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
      if (
          not isinstance(v, core.Literal)
          and v in env
          and (bs := env[v]) != in_block_spec
      ):
        if bs.block_shape != in_block_spec.block_shape:
          in_block_spec = in_block_spec.replace(block_shape=_illegal)
        in_block_spec = in_block_spec.replace(index_map=_illegal)
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

  def _remove_nones(shape: tuple[int | None, ...] | None) -> tuple[int, ...]:
    assert shape is not None
    return tuple(s for s in shape if s is not None)

  _no_aval = object()

  def _get_block_aval(bs, aval):
    if bs is pallas_core.no_block_spec or bs is None:
      return _no_aval
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
    return bs_env[atom]

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
  def new_index_map(i, *args):
    idx = block_spec.index_map(*args)
    assert len(idx) == len(block_spec.block_shape)
    idx = util.tuple_update(idx, i, 0)
    return idx

  new_block_shape = util.tuple_update(block_spec.block_shape, i, 1)
  return pallas_core.BlockSpec(
      new_block_shape, functools.partial(new_index_map, i)
  )


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


def register_binop_rule(prim: core.Primitive):
  register_pull_block_spec_rule(prim)(functools.partial(_binop_pull_rule, prim))
  register_usage_rule(prim)(functools.partial(_binop_usage_rule, prim))
  register_eval_rule(prim)(functools.partial(_binop_eval_rule, prim))


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
register_binop_rule(lax.and_p)
register_binop_rule(ad_util.add_any_p)


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
  assert len(x.shape) == sum(1 for bs in out_block_shape if bs is not None)
  return x


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
  if strides is not None:
    raise NotImplementedError('strides are not supported yet')
  slice_sizes = tuple(
      int(end - start) for start, end in zip(start_indices, limit_indices)
  )
  for bs, slice_start, slice_size in zip(
      block_spec.block_shape, start_indices, slice_sizes
  ):
    if bs is None:
      continue
    assert slice_start % bs == 0, (start_indices, block_spec.block_shape)
    assert slice_size % bs == 0, (slice_sizes, block_spec.block_shape)
  offsets = tuple(
      slice_start // bs if bs is not None else slice_start
      for slice_start, bs in zip(start_indices, block_spec.block_shape)
  )

  def _offset(x, i):
    return x + i if i != 0 else x

  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    assert len(idx) == len(block_spec.block_shape)
    return tuple(_offset(i, o) for i, o in zip(idx, offsets))

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


def _offset(x, i, s):
  from jax.experimental import checkify

  if s is not None:
    pred = i % s == 0
    if isinstance(pred, jax.Array):
      checkify.check(i % s == 0, 'Invalid index', debug=True)
    else:
      if not pred:
        raise ValueError('Invalid index')
  offset = jax.lax.div(i, s) if s is not None else i
  return x + offset


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
  del slice_sizes

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
    # map.
    block_indices = tuple(
        _offset(i, o, s)
        for i, o, s in zip(
            idx, slice_starts, block_spec.block_shape, strict=True
        )
    )
    return block_indices

  new_block_spec = pallas_core.BlockSpec(block_spec.block_shape, new_index_map)
  return [new_block_spec] + [pallas_core.no_block_spec] * (
      len(ctx.avals_in) - 1
  )


@register_eval_rule(lax.concatenate_p)
def _concatenate_eval_rule(ctx: KernelEvalContext, *args, dimension):
  # We now handle the case where each of the concatenated array dimensions
  # divides the block size.
  block_spec = ctx.out_block_specs[0]
  block_shape = block_spec.block_shape
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
  num_blocks = []
  block_dim = block_shape[dimension]
  if block_dim is None:
    block_dim = 1
  if block_dim == sum(aval.shape[dimension] for aval in ctx.avals_in):  # pytype: disable=attribute-error
    # Handle special case if the block contains all of the concatenated
    # array.
    new_shapes = [
        util.tuple_update(
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
    eval_ctx: KernelEvalContext, x, broadcast_dimensions, **params
):
  if not eval_ctx.avals_in[0].shape:  # pytype: disable=attribute-error
    # Scalar -> Array broadcast
    block_spec = eval_ctx.out_block_specs[0]
    shape = tuple(s for s in block_spec.block_shape if s is not None)
    return jax.lax.broadcast_in_dim(x, broadcast_dimensions=(), shape=shape)
  return x


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

  if not ctx.avals_in[0].shape:  # pytype: disable=attribute-error
    return [pallas_core.no_block_spec]

  def new_index_map(*args):
    idx = block_spec.index_map(*args)
    return tuple(idx[i] for i in broadcast_dimensions)

  new_block_shape = tuple(
      block_spec.block_shape[i] for i in broadcast_dimensions
  )
  return [pallas_core.BlockSpec(new_block_shape, new_index_map)]


@register_eval_rule(lax.transpose_p)
def _transpose_eval_rule(
    eval_ctx: KernelEvalContext, x, permutation: tuple[int, ...]
):
  block_spec = eval_ctx.out_block_specs[0]
  block_shape = block_spec.block_shape
  block_shape_no_nones = tuple(bs for bs in block_shape if bs is not None)
  block_dims_iter = iter(range(len(block_shape_no_nones)))
  expanded_block_dims = [
      None if bs is None else next(block_dims_iter) for bs in block_shape
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


@register_eval_rule(lax.iota_p)
def _iota_eval_rule(
    eval_ctx: KernelEvalContext, *, dimension, shape, dtype, sharding
):
  del sharding
  block_spec = eval_ctx.out_block_specs[0]
  block_idx = eval_ctx.get_out_block_indices()[0]
  assert len(block_idx) == len(shape)
  iota_shape = tuple(s for s in block_spec.block_shape if s is not None)
  dim_ = dimension - sum(s is None for s in block_spec.block_shape[:dimension])
  local_iota = jax.lax.broadcasted_iota(dtype, iota_shape, dim_)
  return local_iota + block_idx[dimension] * block_spec.block_shape[dimension]


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


@register_usage_rule(pjit.pjit_p)
def _jit_usage_rule(
    ctx, used_out: list[set[Usage]], *, jaxpr: core.ClosedJaxpr, **_
):
  read_usage_env = compute_usage(jaxpr.jaxpr, used_out)
  in_usages = util.safe_map(read_usage_env, jaxpr.jaxpr.invars)
  return in_usages


@register_eval_rule(pjit.pjit_p)
def _jit_eval_rule(ctx: KernelEvalContext, *args, jaxpr, **kwargs):
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  if consts:
    raise NotImplementedError('pjit with consts not supported yet')
  out_tree = tree_util.tree_structure(tuple(jaxpr.outvars))
  in_tree = tree_util.tree_structure((tuple(jaxpr.invars), {}))
  read_usage_env = compute_usage(jaxpr, ctx.out_usages)
  _, env, _ = _pull_block_spec(
      jaxpr,
      ctx.out_block_specs,
      ctx.out_usages,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
  )
  kernel_fn = make_kernel_function(
      jaxpr,
      in_tree,
      out_tree,
      read_usage_env,
      ctx.in_block_specs,
      env,
      ctx.scalar_prefetch_handler,
      ctx.grid,
  )
  return kernel_fn(ctx.get_program_ids(), ctx.scalar_prefetch, *args)


@register_pull_block_spec_rule(pjit.pjit_p)
def _jit_pull_block_spec_rule(
    ctx: PullRuleContext, out_block_specs, *, jaxpr, **kwargs
):
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  if consts:
    raise NotImplementedError('pjit with consts not supported yet')
  in_block_specs, _, _ = _pull_block_spec(
      jaxpr,
      out_block_specs,
      ctx.out_usages,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
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
  read_usage_env = compute_usage(jaxpr, ctx.out_usages)
  _, env, _ = _pull_block_spec(
      jaxpr,
      ctx.out_block_specs,
      ctx.out_usages,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
  )
  kernel_fn = make_kernel_function(
      jaxpr,
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
  in_block_specs, _, _ = _pull_block_spec(
      jaxpr,
      out_block_specs,
      ctx.out_usages,
      scalar_prefetch_handler=ctx.scalar_prefetch_handler,
      grid=ctx.grid,
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
  if len(block_specs) > 1:
    raise NotImplementedError('select_n with multiple inputs not supported yet')
  return block_specs[0]


@register_push_block_spec_rule(custom_derivatives.custom_jvp_call_p)
def _custom_jvp_call_push_rule(
    ctx, *block_specs, call_jaxpr: core.ClosedJaxpr, **_
):
  assert not call_jaxpr.consts
  return _push_block_spec_jaxpr(call_jaxpr.jaxpr, *block_specs)


@register_push_block_spec_rule(pjit.pjit_p)
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
