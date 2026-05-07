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

import functools
import json

from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import xla_bridge
from jax._src import xla_debug_log_flags
from jax._src.debugging import _split_callback_args
from jax._src.debugging import merge_callback_args
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir

xla_debug_log_flags.set_xla_debug_log_flags()

partial = functools.partial
xla_debug_log_p = core.Primitive("xla_debug_log")
xla_debug_log_p.multiple_results = True
xla_debug_log_p.def_impl(partial(dispatch.apply_primitive, xla_debug_log_p))


class XlaDebugLogEffect(effects.Effect):

  def __str__(self):
    return "XlaDebugLog"


xla_debug_log_effect = XlaDebugLogEffect()

effects.lowerable_effects.add_type(XlaDebugLogEffect)
effects.control_flow_allowed_effects.add_type(XlaDebugLogEffect)
effects.remat_allowed_effects.add_type(XlaDebugLogEffect)
effects.custom_derivatives_allowed_effects.add_type(XlaDebugLogEffect)
effects.partial_eval_kept_effects.add_type(XlaDebugLogEffect)

xla_debug_log_p.def_effectful_abstract_eval(
    lambda *avals, xla_debug_log_kvs, callback=None: (
        [],
        {xla_debug_log_effect},
    )
)


def _xla_debug_log_batching_rule(
    args, dims, *, xla_debug_log_kvs, callback=None
):
  xla_debug_log_p.bind(
      *args, xla_debug_log_kvs=xla_debug_log_kvs, callback=callback
  )
  return [], []


batching.primitive_batchers[xla_debug_log_p] = _xla_debug_log_batching_rule


def _xla_debug_log_jvp_rule(
    primals, tangents, *, xla_debug_log_kvs, callback=None
):
  xla_debug_log_p.bind(
      *primals, xla_debug_log_kvs=xla_debug_log_kvs, callback=callback
  )
  return [], []


ad.primitive_jvps[xla_debug_log_p] = _xla_debug_log_jvp_rule


def _xla_debug_log_transpose_rule(cts, *args, xla_debug_log_kvs, callback=None):
  return [None] * len(args)


ad.primitive_transposes[xla_debug_log_p] = _xla_debug_log_transpose_rule


def log(**kwargs):
  """Sets debug attributes on the XLA op that produces x."""

  kvs = dict(kwargs)
  hashable_metadata = tuple(sorted(kvs.items()))

  def impl(callback, *args, **kwargs):
    in_tree, dyn_args, static_args = _split_callback_args(args, kwargs)

    def _flat_callback(replica_id, partition_id, literals):
      dyn_args_vals = list(literals)
      args_, kwargs_ = merge_callback_args(in_tree, dyn_args_vals, static_args)
      callback(replica_id, partition_id, *args_, **kwargs_)

    xla_debug_log_p.bind(
        *dyn_args, xla_debug_log_kvs=hashable_metadata, callback=_flat_callback
    )

  return impl


def create_hlo_output_callback(callback, hlo_output_callback_id, num_operands):
  """Wraps a python callback as an HLO output callback."""
  try:
    client = xla_bridge.get_backend("tpu")
  except (ValueError, RuntimeError):
    client = xla_bridge.get_backend()
  return client.create_hlo_output_callback(
      callback_id=hlo_output_callback_id,
      num_operands=num_operands,
      callback=callback,
  )


def _xla_debug_log_lowering(ctx, *dyn_args, xla_debug_log_kvs, callback):
  kvs = dict(xla_debug_log_kvs)
  hlo_output_callback_id = ctx.module_context.new_hlo_output_callback_id()
  kvs["callback_id"] = hlo_output_callback_id

  def wrapped_callback(replica_id, partition_id, literals):
    callback(replica_id, partition_id, literals)

  hlo_cb = create_hlo_output_callback(
      wrapped_callback,
      hlo_output_callback_id=hlo_output_callback_id,
      num_operands=len(dyn_args),
  )
  ctx.module_context.add_host_callback(hlo_cb)

  mlir.custom_call(
      "xla.debug.Log",
      result_types=[],
      operands=list(dyn_args),
      has_side_effect=True,
      backend_config=json.dumps({"debug_attributes_config": kvs}),
  )
  return []


# Here we must set cacheable=False, so that each call to xla_debug_log gets a
# unique callback_id and is treated as a distinct host callback.
mlir.register_lowering(
    xla_debug_log_p, _xla_debug_log_lowering, cacheable=False
)
