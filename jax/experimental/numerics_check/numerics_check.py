# Copyright 2024 The JAX Authors.
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
from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Concatenate, NamedTuple, ParamSpec, Protocol, TypeVar

import jax.numpy as jnp
from jax._src import (
  api,
  api_util,
  core,
  custom_derivatives,
  source_info_util,
  traceback_util,
  tree_util,
  typing,
  util,
)
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax

zip = util.safe_zip


Val = Any

class PrecisionDtype(NamedTuple):
  dtype: jnp.dtype
  exponent_bits: int
  mantissa_bits: int


PRECISION_DTYPE_F32 = PrecisionDtype(jnp.float32, 8, 23)
PRECISION_DTYPE_BF16 = PrecisionDtype(jnp.bfloat16, 8, 7)


# Rules


class _NumericsCheckRule(Protocol):
  def __call__(
    self,
    trace: "NumericsCheckTrace",
    in_metrics: tuple[typing.Array, ...],
    out_metric: typing.Array,
    *args: Val,
    **params: Val,
  ) -> tuple[Val, ...]: ...


_numerics_checks: dict[core.Primitive, _NumericsCheckRule] = {}


def register_numerics_check(prim: core.Primitive):
  def register(rule: _NumericsCheckRule):
    _numerics_checks[prim] = rule
    return rule

  return register


# Default Rules


@lru_cache
def _make_default_multiple_results_rule(
  primitive: core.Primitive,
) -> _NumericsCheckRule:
  def default_multiple_results_rule(
    trace: "NumericsCheckTrace",
    in_metrics: tuple[typing.Array, ...],
    out_metric: typing.Array,
    *args: Val,
    **params: Val,
  ) -> tuple[Val, ...]:
    del trace, in_metrics, out_metric
    return primitive.bind(*args, **params)

  return default_multiple_results_rule


def lower_precision(hp: PrecisionDtype, lp: PrecisionDtype, val: Val) -> Val:
  if isinstance(val, typing.Array) and val.dtype == hp.dtype:
    val = lax.reduce_precision(
      val,
      exponent_bits=lp.exponent_bits,
      mantissa_bits=lp.mantissa_bits,
    )
    val = val.astype(lp.dtype)
  return val


@lru_cache
def _make_default_numerics_check(primitive: core.Primitive) -> _NumericsCheckRule:
  @lru_cache
  def make_default_numerics_check_with_kwargs(
    high_precision_dtype: PrecisionDtype,
    low_precision_dtype: PrecisionDtype,
    **params: Val,
  ) -> Val:
    lp = partial(lower_precision, high_precision_dtype, low_precision_dtype)

    @custom_derivatives.custom_vjp
    def default_numerics_check(
      in_metrics: tuple[typing.Array, ...], out_metric: typing.Array, *args: Val
    ) -> Val:
      del in_metrics, out_metric
      return primitive.bind(*args, **params)

    def default_numerics_fwd(
      in_metrics: tuple[typing.Array, ...], out_metric: typing.Array, *args: Val
    ):
      del in_metrics, out_metric
      def bind_primitive(*args):
        return primitive.bind(*args, **params)

      out, f_vjp = api.vjp(
        bind_primitive,
        *args,
      )
      low_precision_out, low_precision_f_vjp = api.vjp(
        lambda *args: lp(bind_primitive(*args)),
        *tuple(map(lp, args)),
      )
      delta = out - low_precision_out.astype(out.dtype)
      return out, (f_vjp, low_precision_f_vjp, delta)

    def default_numerics_bwd(res: tuple[Callable, Callable, Val], g: Val):
      f_vjp, low_precision_f_vjp, delta = res
      out_metric = jnp.sum((g * delta.astype(g.dtype)).astype(jnp.float32))
      grads = f_vjp(g)
      low_precision_grads = low_precision_f_vjp(lp(g))
      in_metrics = tuple(
        jnp.mean(
          jnp.square((grad - low_precision_grad.astype(grad.dtype)).astype(jnp.float32))
        )
        for grad, low_precision_grad in zip(grads, low_precision_grads)
      )
      return (in_metrics, out_metric, *grads)

    default_numerics_check.defvjp(default_numerics_fwd, default_numerics_bwd)
    return default_numerics_check

  def default_numerics_check(
    trace: "NumericsCheckTrace",
    in_metrics: tuple[typing.Array, ...],
    out_metric: typing.Array,
    *args: Val,
    **params: Val,
  ) -> Val:
    return make_default_numerics_check_with_kwargs(
      trace.high_precision_dtype, trace.low_precision_dtype, **params
    )(in_metrics, out_metric, *args)

  return default_numerics_check


@lru_cache
def _make_dupe(
  high_precision_dtype: PrecisionDtype, low_precision_dtype: PrecisionDtype
) -> Callable[[typing.Array, Val], tuple[Val, Val]]:
  lp = partial(lower_precision, high_precision_dtype, low_precision_dtype)

  @custom_derivatives.custom_vjp
  def _dupe(in_metric: typing.Array, arg: Val) -> tuple[Val, Val]:
    del in_metric
    return arg, arg

  def _dupe_fwd(in_metric: typing.Array, arg: Val) -> tuple[tuple[Val, Val], None]:
    return _dupe(in_metric, arg), None

  def _dupe_bwd(
    res: tuple[tuple[Val, Val], None], g: tuple[Val, Val]
  ) -> tuple[Val, Val]:
    del res
    g1, g2 = g
    grad = g1 + g2
    low_precision_grad = lp(g1) + lp(g2)
    in_metric = jnp.mean(
      jnp.square((grad - low_precision_grad.astype(grad.dtype)).astype(jnp.float32))
    )
    return in_metric, grad

  _dupe.defvjp(_dupe_fwd, _dupe_bwd)
  return _dupe


# Trace


class NumericsCheckTracer(core.Tracer):
  _trace: "NumericsCheckTrace"
  val: Val

  def __init__(self, trace, val):
    self._trace = trace
    self.val = val

  @property
  def aval(self) -> core.AbstractValue:
    return core.get_aval(self.val)

  def to_concrete_value(self) -> Val:
    return core.to_concrete_value(self.val)

class _NumericsCheckTracerAsName:
  ref: Val
  tracer: NumericsCheckTracer

  def __init__(self, tracer: NumericsCheckTracer):
    self.ref = core.get_referent(tracer)
    self.tracer = tracer

  def __eq__(self, other):
    return isinstance(other, _NumericsCheckTracerAsName) and self.ref is other.ref

  def __hash__(self):
    return id(self.ref)


class MetricsKey:
  primitive: core.Primitive
  source_info: source_info_util.SourceInfo
  in_metrics: int
  uses: int
  in_avals: tuple[api.ShapeDtypeStruct, ...]

  def __init__(
    self,
    primitive: core.Primitive,
    source_info: source_info_util.SourceInfo,
    in_metrics: int,
    in_avals: tuple[api.ShapeDtypeStruct, ...],
  ):
    self.primitive = primitive
    self.source_info = source_info
    self.in_metrics = in_metrics
    self.uses = 0
    self.in_avals = in_avals

  # It's slightly faster to use a class with __slots__ than a NamedTuple.
  __slots__ = ["primitive", "source_info", "in_metrics", "uses", "in_avals"]


class MetricKeys:
  keys: list[MetricsKey]

  def __init__(self, keys: list[MetricsKey]):
    self.keys = keys

  def replace(self, **kwargs) -> MetricKeys:
    """Returns a new MetricKeys with the specified fields replaced."""
    new_kwargs = {field: getattr(self, field) for field in self.__slots__}
    new_kwargs.update(kwargs)
    return MetricKeys(**new_kwargs)

  # It's slightly faster to use a class with __slots__ than a NamedTuple.
  __slots__ = ["keys"]


MetricsValue = tuple[tuple[typing.Array, ...], typing.Array, tuple[typing.Array, ...]]
Metrics = list[MetricsValue]


class _DupeInfo:
  metric_key: MetricsKey
  dupes: list[NumericsCheckTracer]

  def __init__(self, metric_key: MetricsKey):
    self.metric_key = metric_key
    self.dupes = []

  __slots__ = ["metric_key", "dupes"]


TracerToDupeInfo = dict[_NumericsCheckTracerAsName, _DupeInfo]


class NumericsCheckTrace(core.Trace[NumericsCheckTracer]):
  parent_trace: core.Trace
  tag: core.TraceTag
  high_precision_dtype: PrecisionDtype
  low_precision_dtype: PrecisionDtype
  metric_keys: MetricKeys

  next_metric_index: int
  tracer_to_dupe_info: TracerToDupeInfo
  metrics: None | Metrics

  def __init__(
    self,
    parent_trace,
    tag,
    high_precision_dtype: PrecisionDtype,
    low_precision_dtype: PrecisionDtype,
    next_metric_index: int,
    tracer_to_dupe_info: TracerToDupeInfo,
    metrics: None | Metrics,
  ):
    self.parent_trace = parent_trace
    self.tag = tag
    self.high_precision_dtype = high_precision_dtype
    self.low_precision_dtype = low_precision_dtype
    self.metric_keys = MetricKeys([])
    self.next_metric_index = next_metric_index
    self.tracer_to_dupe_info = tracer_to_dupe_info
    self.metrics = metrics

  def to_val(self, val: Val | NumericsCheckTracer) -> Val:
    if isinstance(val, NumericsCheckTracer) and val._trace.tag is self.tag:
      return val.val
    else:
      return val

  @staticmethod
  def make_metric() -> typing.Array:
    return jnp.zeros((), dtype=jnp.float32)

  def process_primitive(
    self, primitive: core.Primitive, tracers: tuple[Val, ...], params: dict[str, Val]
  ) -> Val:
    duped_in_tracers = []
    for in_tracer in tracers:
      if dupe_info := self.tracer_to_dupe_info.get(
        _NumericsCheckTracerAsName(in_tracer)
      ):
        dupe_info.metric_key.uses += 1
        if dupe_info.dupes:
          in_tracer = dupe_info.dupes.pop(0)
      duped_in_tracers.append(in_tracer)
    tracers = tuple(duped_in_tracers)

    rule = _numerics_checks.get(primitive, None)
    if rule is None:
      if primitive.multiple_results:
        rule = _make_default_multiple_results_rule(primitive)
      else:
        rule = _make_default_numerics_check(primitive)
    in_vals = tuple(map(self.to_val, tracers))

    metrics: MetricsValue | tuple[tuple[None, ...], None, tuple[None, ...]]
    if self.metrics is None:
      metrics = ((None,) * len(in_vals), None, ())
    else:
      metrics = self.metrics[self.next_metric_index]
      self.next_metric_index += 1
    in_metrics = tuple(
      NumericsCheckTrace.make_metric() if metric is None else metric
      for metric in metrics[0]
    )
    out_metric = NumericsCheckTrace.make_metric() if metrics[1] is None else metrics[1]
    self.metric_keys.keys.append(
      MetricsKey(
        primitive,
        source_info_util.current(),
        len(in_metrics),
        tuple(api.ShapeDtypeStruct(x.shape, x.dtype) for x in in_vals),
      )
    )

    with core.set_current_trace(self.parent_trace):
      out_vals = rule(self, in_metrics, out_metric, *in_vals, **params)
    if primitive.multiple_results:
      out_tracers = tuple(map(partial(NumericsCheckTracer, self), out_vals))
    else:
      out_tracers = (NumericsCheckTracer(self, out_vals),)

    for out_tracer in out_tracers:
      dupe_info = _DupeInfo(self.metric_keys.keys[-1])
      self.tracer_to_dupe_info[_NumericsCheckTracerAsName(out_tracer)] = dupe_info
      if self.metrics is not None and len(metrics[2]) > 0:
        dupe = _make_dupe(self.high_precision_dtype, self.low_precision_dtype)
        next_copy = out_tracer
        for dupe_metric in metrics[2]:
          with core.set_current_trace(self.parent_trace):
            copy, next_copy = tuple(
              map(
                partial(NumericsCheckTracer, self),
                dupe(dupe_metric, self.to_val(next_copy)),
              )
            )
          dupe_info.dupes.append(copy)
        dupe_info.dupes.append(next_copy)
    return tuple(out_tracers) if primitive.multiple_results else out_tracers[0]


# Transformation


P = ParamSpec("P")
R = TypeVar("R")


@lu.transformation_with_aux2
def numerics_check_subtrace(
  f: Callable,
  store: lu.Store,
  tag: core.TraceTag,
  high_precision_dtype: PrecisionDtype,
  low_precision_dtype: PrecisionDtype,
  next_metric_index: int,
  tracer_to_dupe_info: TracerToDupeInfo,
  metrics: Metrics,
  *args_flat: Val,
) -> tuple[Val, ...]:
  with core.take_current_trace() as parent_trace:
    trace = NumericsCheckTrace(
      parent_trace,
      tag,
      high_precision_dtype,
      low_precision_dtype,
      next_metric_index,
      tracer_to_dupe_info,
      metrics,
    )
    in_tracers = tuple(map(partial(NumericsCheckTracer, trace), args_flat))
    with core.set_current_trace(trace):
      out_tracers = f(*in_tracers)
    out = tuple(map(trace.to_val, out_tracers))
    store.store(dict(trace=trace))
  return out


@lu.transformation2
def numerics_check_trace(
  f: Callable[
    Concatenate[
      core.TraceTag,
      PrecisionDtype,
      PrecisionDtype,
      int,
      TracerToDupeInfo,
      Metrics,
      ...,
    ],
    R,
  ],
  subtrace_thunk: Callable,
  high_precision_dtype: PrecisionDtype,
  low_precision_dtype: PrecisionDtype,
  metrics: Metrics,
  *args_flat: Val,
) -> R:
  tag = core.TraceTag()
  with source_info_util.transform_name_stack("numerics_check"):
    out = f(tag, high_precision_dtype, low_precision_dtype, 0, {}, metrics, *args_flat)
  trace = subtrace_thunk()["trace"]
  with core.ensure_no_leaks(trace):
    del trace
  return out


def numerics_check(
  fun: Callable[P, R],
  high_precision_dtype: PrecisionDtype = PRECISION_DTYPE_F32,
  low_precision_dtype: PrecisionDtype = PRECISION_DTYPE_BF16,
) -> tuple[
  Callable[Concatenate[Metrics, P], Val],
  Callable[P, MetricKeys],
]:
  api_util.check_callable(fun)
  docstr = "Takes similar arguments as {fun} but adds additional arrays in which numerical sensitivities are deposited."
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  @util.wraps(fun, docstr=docstr)
  @traceback_util.api_boundary
  def numerics_check_f(
    metrics: Metrics,
    *args: P.args,
    **kwargs: P.kwargs,
  ) -> Val:
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    f = lu.wrap_init(fun)
    f, out_tree_thunk = api_util.flatten_fun(f, in_tree)
    f, subtrace_thunk = numerics_check_subtrace(f)
    f = numerics_check_trace(
      f, subtrace_thunk, high_precision_dtype, low_precision_dtype, metrics
    )
    out_flat = f.call_wrapped(*args_flat)
    return tree_util.tree_unflatten(out_tree_thunk(), out_flat)

  def numerics_check_metrics_f(
    *args: P.args,
    **kwargs: P.kwargs,
  ) -> MetricKeys:
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    f = lu.wrap_init(fun)
    f, _ = api_util.flatten_fun(f, in_tree)
    f, subtrace_thunk = numerics_check_subtrace(f)
    f = numerics_check_trace(
      f, subtrace_thunk, high_precision_dtype, low_precision_dtype, None
    )
    pe.trace_to_jaxpr_dynamic(f, tuple(core.get_aval(x) for x in args_flat))
    return subtrace_thunk()["trace"].metric_keys

  return numerics_check_f, numerics_check_metrics_f


def metric_keys_to_metrics(metric_keys: MetricKeys) -> Metrics:
  return [
    (
      tuple(NumericsCheckTrace.make_metric() for _ in range(key.in_metrics)),
      NumericsCheckTrace.make_metric(),
      tuple(NumericsCheckTrace.make_metric() for _ in range(key.uses - 1)),
    )
    for key in metric_keys.keys
  ]

def sort_metrics_by_in_metrics(
  metric_keys: MetricKeys, metrics: Metrics
) -> tuple[MetricKeys, Metrics]:
  def sort_key(key_and_metric: tuple[MetricsKey, MetricsValue]) -> int:
    _, (in_metrics, _, _) = key_and_metric
    return max(abs(x.item()) for x in in_metrics)

  sorted_pairs = sorted(zip(metric_keys.keys, metrics), key=sort_key, reverse=True)
  return metric_keys.replace(keys=[x[0] for x in sorted_pairs]), [
    x[1] for x in sorted_pairs
  ]


def sort_metrics_by_out_metric(
  metric_keys: MetricKeys, metrics: Metrics
) -> tuple[MetricKeys, Metrics]:
  def sort_key(key_and_metric: tuple[MetricsKey, MetricsValue]) -> int:
    _, (_, out_metric, _) = key_and_metric
    return abs(out_metric.item())

  sorted_pairs = sorted(zip(metric_keys.keys, metrics), key=sort_key, reverse=True)
  return metric_keys.replace(keys=[x[0] for x in sorted_pairs]), [
    x[1] for x in sorted_pairs
  ]


def sort_metrics_by_dupe_metrics(
  metric_keys: MetricKeys, metrics: Metrics
) -> tuple[MetricKeys, Metrics]:
  def sort_key(key_and_metric: tuple[MetricsKey, MetricsValue]) -> int:
    _, (_, _, dupe_metrics) = key_and_metric
    return sum(abs(x.item()) for x in dupe_metrics)

  sorted_pairs = sorted(zip(metric_keys.keys, metrics), key=sort_key, reverse=True)
  return metric_keys.replace(keys=[x[0] for x in sorted_pairs]), [
    x[1] for x in sorted_pairs
  ]

def print_metrics(
  metric_keys: MetricKeys, metrics: Metrics, *, normalize_out_metric: bool = False
):
  out_metrics = [m[1].item() for m in metrics]
  if normalize_out_metric:
    normalizer = max(abs(m) for m in out_metrics)
    out_metrics = [m / normalizer for m in out_metrics]

  for key, (in_metrics, _, dupe_metrics), out_metric in zip(
    metric_keys.keys, metrics, out_metrics
  ):
    print(f"\n{key.primitive}:{source_info_util.summarize(key.source_info)}:")
    print(f"  In avals: {key.in_avals}")
    print(f"  In metrics: {in_metrics}")
    print(f"  Out metric: {out_metric}")
    print(f"  Dupe metrics: {dupe_metrics}")
