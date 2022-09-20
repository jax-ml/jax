import dataclasses

from typing import Any, Dict

from jax import core
from jax import tree_util
from jax.interpreters import mlir
from jax._src.lib import xla_bridge as xb
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo

# from jax._src.lib import custom_call_lib  # TODO: implement this

custom_call_p = core.Primitive("jax_custom_call")
custom_call_p.multiple_results = True

def _custom_call_abstract_eval(*avals, out_avals, name, **_):
  del avals
  return out_avals
custom_call_p.def_abstract_eval(_custom_call_abstract_eval)

def _custom_call_lowering_rule(ctx: mlir.LoweringRuleContext, *in_nodes,
                               out_avals, name, descriptor):
  custom_call_descriptor = "" # TODO: implement the following API
  keepalive = None
  # custom_call_descriptor, keepalive = custom_call_lib.make_descriptor(name, descriptor)

  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_aval.shape, mlir.dtype_to_ir_type(out_aval.dtype))
      for out_aval in out_avals])
  i32_type = ir.IntegerType.get_signless(32)
  ctx.module_context.add_keepalive(keepalive)
  out = mhlo.CustomCallOp(
            [out_type], in_nodes,
            call_target_name=ir.StringAttr.get("jax_custom_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(custom_call_descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 2),
            called_computations=ir.ArrayAttr.get([]))
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_avals))]
  return results
mlir.register_lowering(custom_call_p, _custom_call_lowering_rule, platform="cpu")

@dataclasses.dataclass
class CustomCall:
  name: str
  _registry: Dict[str, Any] = dataclasses.field(init=False,
                                                default_factory=dict)

  def register(self, function_ptr, *, platform):
    platform = platform.upper()
    if platform != "CPU":
      raise NotImplementedError(platform)
    # TODO: register function pointer our C++ registry

  def __call__(self, *args, out_shape_dtype, descriptor=None):
    flat_out_shape_dtype, out_tree = tree_util.tree_flatten(out_shape_dtype)
    out_avals = [core.ShapedArray(a.shape, a.dtype)
                 for a in flat_out_shape_dtype]
    out_flat = custom_call_p.bind(*args, out_avals=out_avals,
        descriptor=descriptor, name=self.name)
    return tree_util.tree_unflatten(out_tree, out_flat)

def custom_call(name: str):
  return CustomCall(name)
