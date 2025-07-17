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

from functools import partial
from typing import Any

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import tree_util
from jax._src import xla_metadata_lib
from jax._src.interpreters import ad, batching, mlir
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir

config_ext = xla_client._xla.config


class XlaMetadataContextManager:
  __slots__ = ["prev", "updates"]

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        xla_metadata_lib.update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)


def set_xla_metadata(x=None, **kwargs):
  if x is None:
    return XlaMetadataContextManager(kwargs)
  else:
    hashable_metadata = tuple(sorted(kwargs.items()))
    return tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(v, xla_metadata_kvs=hashable_metadata),
        x,
    )


# `xla_metadata_value_p` is an identity primitive for attaching frontend_attributes
# to the primitive's producing (parent/owner) op.
xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(partial(dispatch.apply_primitive, xla_metadata_value_p))
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata_kvs: aval)
batching.defvectorized(xla_metadata_value_p)
# TODO(nbasile): Implement tagging gradient ops with metadata.
ad.deflinear2(xla_metadata_value_p, lambda ct, _: (ct,))

def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, val: ir.Value, *, xla_metadata_kvs):
  xla_metadata = dict(xla_metadata_kvs)
  op_to_attach_metadata = _target_op_to_attach_metadata(val)
  if op_to_attach_metadata is not None:
    _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
  return [val]

# If we leave `cacheable=True`, when we are in the lowering rule, the `val.owner`
# becomes a cached `FuncOp`. FuncOp.owners are Blocks, which we can't tag.
mlir.register_lowering(
    xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False)


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
  op = value_mlir.owner
  if op is None or isinstance(op, ir.Block):
    return None
  # TODO(nbasile): Add logic for handling multiply-by-constant-1.0 ops, which
  # are often added by jax gradients.
  # [Couple this change with tagging gradient ops.]
  return op


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  if xla_metadata:
    ctx_attributes, existing_attributes = {}, {}
    for k, v in xla_metadata.items():
      ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
    # Combine with existing mhlo.frontend_attributes
    for attr in op.attributes:
      if attr.name == "mhlo.frontend_attributes":
        for a in attr.attr:
          existing_attributes[a.name] = a.attr
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        ctx_attributes | existing_attributes
    )
