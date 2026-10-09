# Copyright 2026 The JAX Authors.
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

"""Rebindable: kernel calls with rebindable static hyperparameters."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
import dataclasses
from functools import partial, update_wrapper
import inspect
import operator
import threading
from typing import Any

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import flattree as ft
from jax._src import stages
from jax._src import traceback_util
from jax._src.api_util import debug_info
from jax._src.core import typeof
from jax._src.errors import UnexpectedTracerError
from jax._src.hijax import HiPrim
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (
    tracing_registry, tree_leaves, tree_leaves_checked, tree_unflatten)
from jax._src.util import fun_name, weakref_lru_cache

traceback_util.register_exclusion(__file__)

def _typeof_tree(args):
  leaves, tree = tracing_registry.flatten(args)
  return tree_unflatten(tree, [typeof(x) for x in leaves])


class _BodyStack(threading.local):
  names: tuple[str, ...] = ()  # rebindables whose bodies are being traced

_bodies = _BodyStack()


# Keyed on a Rebindable's identity, including the trace context it was first
# bound in instead of the ambient one, so lowering (in any context) reuses that
# body.
@partial(weakref_lru_cache, trace_context_in_key=False)
def _trace_cached(fn, in_tree, in_avals_flat, hp_items, trace_context):
  del trace_context  # part of the key only
  in_avals, hyperparams = tree_unflatten(in_tree, in_avals_flat), dict(hp_items)
  static = tuple(hyperparams)
  prev, _bodies.names = _bodies.names, (*_bodies.names, fun_name(fn))
  args = ft.flatten_static_argnums_argnames(in_avals, hyperparams, (), static)
  dbg = debug_info("rebindable", fn, in_avals, hyperparams,
                   static_argnames=static)
  try:
    jaxpr, out_avals = pe.trace_to_jaxpr(fn, args, dbg)
  finally:
    _bodies.names = prev
  if any(isinstance(c, core.Tracer) for c in jaxpr.consts):
    raise UnexpectedTracerError(
        f"rebindable {fun_name(fn)} closes over a traced value; pass it as an "
        "explicit argument instead.")
  return jaxpr, out_avals


class Rebindable(HiPrim):
  """An application of `fn(*args, **hyperparams)` with rebindable hyperparams.

  Hyperparameters are static and must not change the output types, so
  `rebind` is O(1): the body is traced again only when the rebound application
  is lowered, inside its original context (mesh, axis env). Transformations
  that map over `fn` (vmap, fuser physicalization) keep it a Rebindable. To
  remat a Rebindable is one opaque operation: remat2 asks the policy about the
  whole call, remat3 recomputes it (as it does any operation without a remat
  rule).
  """
  fn: Callable
  name: str
  key: Hashable
  trace_context: Any  # config.trace_context() when first bound

  def __init__(self, fn, hyperparams, in_avals, *, name, key=None,
               trace_context=None, out_type=None):
    # `out_type` is `(out_aval, effects)` if already known, e.g. when rebinding.
    for k, v in hyperparams.items():
      try:
        hash(v)
      except TypeError:
        raise TypeError(
            f"{name} hyperparameter {k}={v!r} is not hashable") from None
    self.in_avals = tuple(in_avals)
    if trace_context is None:
      trace_context = config.trace_context()
    hp_items = tuple(sorted(hyperparams.items()))
    if out_type is None:
      leaves, tree = tracing_registry.flatten(self.in_avals)
      jaxpr, out_avals = _trace_cached(fn, tree, tuple(leaves), hp_items,
                                       trace_context)
      out_type = out_avals.unflatten(), core.positional_effects(jaxpr)
    self.out_aval, self.effects = out_type[0], frozenset(out_type[1])
    self.params = dict(fn=fn, hyperparams=hp_items,
                       name=name, key=key,
                       trace_context=trace_context, in_avals=self.in_avals)
    super().__init__()

  def _new(self, fn, hyperparams, in_avals, out_type=None) -> Rebindable:
    return Rebindable(fn, hyperparams, in_avals, name=self.name,
                   key=self.key,
                   trace_context=self.trace_context, out_type=out_type)

  hyperparams = property(lambda self: dict(self.params["hyperparams"]))

  def pp_params(self):
    return dict(name=self.name, **self.hyperparams)

  def rebind(self, **hyperparams) -> Rebindable:
    if unknown := set(hyperparams) - set(self.hyperparams):
      raise ValueError(f"{self.name} has no hyperparameters {sorted(unknown)}")
    return self._new(self.fn, {**self.hyperparams, **hyperparams},
                     self.in_avals, (self.out_aval, self.effects))

  @property
  def jaxpr(self) -> core.Jaxpr:
    """The body, traced (and cached) for the current hyperparameters."""
    jaxpr, out_avals = _trace_cached(
        self.fn, self.in_tree, tuple(self.in_avals_flat),
        self.params["hyperparams"], self.trace_context)
    effects = frozenset(core.positional_effects(jaxpr))
    if (list(out_avals) != self.out_avals_flat
        or out_avals.tree != self.out_tree or effects != self.effects):
      raise TypeError(
          f"{self.name} output types and effects must not depend on its "
          f"hyperparameters, but {self.hyperparams} changed them from "
          f"{self.out_aval}, {set(self.effects)} to {out_avals.unflatten()}, "
          f"{set(effects)}")
    return jaxpr

  def expand(self, *args):
    jaxpr = self.jaxpr
    out = core.eval_jaxpr(jaxpr, jaxpr.consts,
                          *tree_leaves_checked(self.in_tree, args))
    return tree_unflatten(self.out_tree, out)

  def _map_fn(self, transform, *args):
    return self._new(transform(self.fn), self.hyperparams,
                     _typeof_tree(args))(*args)

  def batch(self, axis_data, args, dims):
    _, out_dims = batching.batch_jaxpr2(
        self.jaxpr, axis_data, tuple(self.in_tree.flatten_up_to(dims)))
    out_dims = tree_unflatten(self.out_tree, out_dims)
    vmap = lambda fn: lambda *xs, **hp: api.vmap(
        partial(fn, **hp), in_axes=dims, out_axes=out_dims,
        axis_size=axis_data.size, axis_name=axis_data.name,
        spmd_axis_name=axis_data.spmd_name or axis_data.explicit_mesh_axis)(*xs)
    return self._map_fn(vmap, *args), out_dims

  def physicalize(self, ctx, *args_flat):
    phys = lambda fn: ctx.physicalize(
        fn, static_argnames=tuple(self.hyperparams))
    return tree_leaves(
        self._map_fn(phys, *tree_unflatten(self.in_tree, args_flat)))

  def _no_ad(self, *_, **__):
    # Residuals of a differentiated body generally depend on the
    # hyperparameters (e.g. tile-sized metadata), which would break rebinding.
    raise NotImplementedError(
        f"rebindable {self.name} is not differentiable; call it from the rules "
        "of a jax.custom_vjp so forward and backward kernels are separate "
        "rebindables.")
  jvp = lin = vjp_fwd = transpose = _no_ad


def rebindable(fn: Callable | None = None, *, hyperparams: str | Sequence[str],
            name: str | None = None, key: Hashable = None):
  """Stages calls `fn(*args, **kwargs)` as :class:`Rebindable` applications.

  Positional arguments are operands: pytrees of arrays, or fuser `Fusion`
  objects, so a rebindable called inside a `fuser.fusible` body takes the fused
  prologue and epilogue as inputs. The keyword arguments named in `hyperparams`
  are the hashable, static values a tuner may rebind (defaults in `fn`'s
  signature are recorded when not passed). Any other keyword arguments are
  fixed static configuration: bound to `fn` as-is, never hashed or exposed for
  rebinding. Hyperparameter-dependent ops written in the body run inside the
  kernel's trace but are not fused as prologue/epilogue; build such a `Fusion`
  explicitly if needed. `key` is an optional hashable label for the call's
  sites, e.g. to rebind them from a table of tuned values.
  """
  if fn is None:
    return partial(rebindable, hyperparams=hyperparams, name=name, key=key)
  hp_names = ((hyperparams,) if isinstance(hyperparams, str)
              else tuple(hyperparams))
  sig = inspect.signature(fn).parameters
  defaults = {k: sig[k].default for k in hp_names
              if k in sig and sig[k].default is not inspect.Parameter.empty}
  op_name = name or fun_name(fn)

  @traceback_util.api_boundary
  def wrapper(*args, **kwargs):
    if _bodies.names:
      raise ValueError(
          f"rebindable {op_name} is called inside the body of rebindable "
          f"{_bodies.names[-1]}; nested rebindables are not supported. Make "
          f"{op_name} a plain function of the outer rebindable's "
          "hyperparameters.")
    hp = {**defaults, **{k: kwargs.pop(k) for k in hp_names if k in kwargs}}
    if missing := set(hp_names) - hp.keys():
      raise TypeError(f"{op_name} missing hyperparameters {sorted(missing)}")
    f = partial(fn, **kwargs) if kwargs else fn
    return Rebindable(f, hp, _typeof_tree(args), name=op_name,
                   key=key)(*args)
  return update_wrapper(wrapper, fn)


@dataclasses.dataclass(frozen=True)
class RebindableSite:
  """A call of a rebindable in a traced program.

  `rebindable` is what runs (function, hyperparameters, operand types, key) and
  `ctx` the context it was called in (abstract mesh, xla_metadata, ...).
  """
  rebindable: Rebindable
  ctx: core.JaxprEqnContext
  _eqn: core.JaxprEqn = dataclasses.field(repr=False)

  @property
  def name(self) -> str:
    return self.rebindable.name

  @property
  def key(self) -> Hashable:
    return self.rebindable.key

  @property
  def abstract_mesh(self):
    return self.ctx.cur_abstract_mesh

  def call(self, *args_flat, **hyperparams):
    """Applies this site's kernel to new operands, e.g. to benchmark it alone.

    `args_flat` are arrays (or refs) typed like
    `self.rebindable.in_avals_flat`, which `self.rebindable.in_tree` rebuilds
    into the original operands, including fused prologues and epilogues. For a
    site inside a `shard_map` (manual axes in `self.abstract_mesh`) these are
    the per-shard operands, so the call must run where each shard sees exactly
    those types: under `jax.jit` for a site outside any mesh, or under a
    `shard_map` over a mesh with the site's manual axis names. Fused prologues
    and epilogues are replayed from jaxprs recorded under the site's mesh, so a
    site with them needs a mesh of that shape: to tune it on a smaller mesh,
    trace the program on that mesh and use its matching site.
    `hyperparams` update the site's.

    The context is the caller's: wrap the call in `jax.jit` (and `shard_map`),
    apply the call's metadata with `set_xla_metadata(**(self.ctx.xla_metadata
    or {}))`, and create refs for in-place operands.
    """
    t = self.rebindable.rebind(**hyperparams)
    return t.fn(*tree_unflatten(t.in_tree, args_flat), **t.hyperparams)

  @property
  def jaxpr(self) -> core.Jaxpr:
    """The body, traced inside the call's mesh and manual axes."""
    mesh, t = self.abstract_mesh, self.rebindable
    with (self.ctx.manager, config._check_vma(_checks_vma(t)),
          core.extend_axis_env_nd(
              [(n, mesh.shape[n]) for n in mesh.manual_axes])):
      return t.jaxpr


def _checks_vma(p: Rebindable) -> bool:
  # Varying-axis types only appear when the enclosing shard_map checks them.
  return any(
      getattr(getattr(a, "inner_aval", a), "manual_axis_type",
              core.empty_mat).varying
      for a in [*p.in_avals_flat, *p.out_avals_flat])


def _map_rebindables(jaxpr: core.Jaxpr, sites: list[RebindableSite],
                     rebind=None) -> core.Jaxpr:
  """Appends each nested Rebindable's site to `sites` and returns `jaxpr` with
  it replaced by `rebind(site)`, if given and not None.

  Unchanged sub-jaxprs, eqns and HiPrims are returned as the same objects.
  """
  def sub(v, eqn):
    if isinstance(v, core.Jaxpr):
      return _map_rebindables(v, sites, rebind)
    if isinstance(v, stages.Traced):  # e.g. CustomVJPTraced's primal
      new = _map_rebindables(v.jaxpr, sites, rebind)
      return v if new is v.jaxpr else v.replace_jaxpr(new)
    if isinstance(v, tuple):
      new = tuple(sub(x, eqn) for x in v)
      return v if all(map(operator.is_, new, v)) else new
    if isinstance(v, Rebindable):
      sites.append(site := RebindableSite(v, eqn.ctx, eqn))
      return (rebind and rebind(site)) or v
    if isinstance(v, HiPrim):  # e.g. Fusible, RematTraced, VmapOf
      changed = {k: new for k, x in v.params.items()
                 if (new := sub(x, eqn)) is not x}
      # A type-preserving rewrite: rebinding never changes types or effects.
      return v.replace(**changed) if changed else v
    return v

  eqns = []
  for eqn in jaxpr.eqns:
    params = {k: sub(v, eqn) for k, v in eqn.params.items()}
    same = all(map(operator.is_, params.values(), eqn.params.values()))
    eqns.append(eqn if same else eqn.replace(params=params))
  if all(map(operator.is_, eqns, jaxpr.eqns)):
    return jaxpr
  return jaxpr.replace(eqns=eqns)


def extract_rebindables(traced: stages.Traced) -> list[RebindableSite]:
  """Returns every Rebindable application in a traced program, in order.

  Descends into sub-jaxprs and HiPrims (e.g. fusibles, remat, vmap). Rebindables
  called from the fwd/bwd rules of a `jax.custom_vjp` appear only in a program
  traced under differentiation (e.g. `jax.grad`), since the rules run only
  then. Rebindables themselves are never differentiated: doing so raises
  `NotImplementedError`. Calling a rebindable inside another rebindable's body
  raises a `ValueError`.
  """
  sites: list[RebindableSite] = []
  _map_rebindables(traced.jaxpr, sites)
  return sites


def rebind(traced: stages.Traced, rule) -> stages.Traced:
  """Returns `traced` with Rebindables rebound per `rule`, without retracing.

  `rule` maps a site `key` (see `rebindable`) to new hyperparameters, or is a
  callable `RebindableSite -> dict | None`; sites it maps to None are unchanged.
  Rebound bodies are traced when the result is lowered, inside the original
  context (mesh, axis env) of each site.
  """
  if not callable(rule) and not rule:
    return traced
  get = rule if callable(rule) else (
      lambda site: None if site.key is None else rule.get(site.key))
  rebind_site = lambda site: (hp := get(site)) and site.rebindable.rebind(**hp)
  new = _map_rebindables(traced.jaxpr, [], rebind_site)
  return traced if new is traced.jaxpr else traced.replace_jaxpr(new)
