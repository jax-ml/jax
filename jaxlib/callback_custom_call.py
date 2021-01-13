# Copyright 2021 Google LLC
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
"""Python API for emitting a CustomCall op for host_callback.

The main entry point is `emit_custom_call`, called from `host_callback.py`
with the `callback_id` and the following operands:

 * an input token, to ensure proper op dependencies,
 * the current replica id of the calling device,
 * the actual arrays to pass to the callback, allocated on the calling device.

The `callback_id` and the shapes of the arguments and results are serializied
into a Descriptor bytearray using Flatbuffer (see callback_custom_call.fbs for
the buffer description, and callback_custom_call.cc for the encoding/decoding
logic.)

For CPU the descriptor array is passed as the very first operand (before the
input token), and for GPU it is passed as the `opaque` string.

The CPU implementation of the `callback_custom_call` is in
`callback_custom_call_cpu_py.cc` and for GPU it is in
`callback_custom_call_cuda_py.cc`. Those functions decode the Descriptor
argument, and eventually call `RunHostCallback` defined in
`callback_custom_call.cc`.
On the GPU, the input arrays are copied first to the host, and the results
are copied back to the device.

The actual invocation of the Python callback is done by `RunHostCallback`, by
way of a trampoline: a global function pointer set by the host_callback.py
upon initialization using the `set_callback_trampoline` entry point.
It is important for the Python code to set the trampoline to `None` when
exiting.

"""
from typing import Any, Callable, Optional, Tuple
import numpy as np

from jaxlib import xla_client

from . import callback_custom_call_cpu_py as callback_cpu  # type: ignore[import-error]
for _name, _value in callback_cpu.custom_call_registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

try:
  from . import callback_custom_call_cuda_py as callback_cuda  # type: ignore[import-error]
  for _name, _value in callback_cuda.custom_call_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
  pass  # We may have built without CUDA support

XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any  # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder


def set_callback_trampoline(trampoline: Optional[Callable]):
  callback_cpu.set_callback_trampoline(trampoline)


def emit_custom_call(platform: str, comp: XlaComputationBuilder,
                     callback_id: int, operands: Tuple[XlaOp, ...],
                     flat_results_xla_shapes: Tuple[XlaShape, ...], *,
                     ignore_results: bool) -> Tuple[XlaOp, ...]:
  """Emits the CustomCall to the given computation.

  Args:
    platform: the platform for which we are generating code, e.g., "cpu".
    comp: the computation into which to emit the ops.
    callback_id
    operands: a tuple containing: the input token, the replica id, and the
      arrays to pass to the callback.
    flat_results_xla_shapes: the expected shapes of results, including the
      output token at the start.
    ignore_results: whether the calling code ignores the callback results.
  """
  operand_shapes = tuple([
      _shape_with_default_layout(comp.GetShape(arg_op)) for arg_op in operands
  ])

  result_shapes = tuple([
      _shape_with_default_layout(res_shape)
      for res_shape in flat_results_xla_shapes
  ])

  descriptor_bytes = callback_cpu.encode_descriptor(
      callback_id, operand_shapes, ignore_results, result_shapes)

  if platform == "cpu":
    # Prepend the serialized descriptor.
    descriptor_op: XlaOp = xla_client.ops.Constant(
        comp, np.array(list(descriptor_bytes), np.uint8))
    operands = (descriptor_op,) + operands
    operand_shapes = (
        (_shape_with_default_layout(comp.GetShape(descriptor_op)),) +
        operand_shapes)
    opaque = b""
  elif platform == "gpu":
    opaque = bytes(descriptor_bytes)
  else:
    raise NotImplementedError(
        f"CustomCall callback not implemented for {platform}")

  call_results = xla_client.ops.CustomCallWithLayout(
      comp,
      b"callback_custom_call",
      operands=operands,
      shape_with_layout=xla_client.Shape.tuple_shape(result_shapes),
      operand_shapes_with_layout=operand_shapes,
      opaque=opaque,
      # The side_effect prevents the op from disappearing when there are no
      # results.
      has_side_effect=True)

  return tuple([
      xla_client.ops.GetTupleElement(call_results, i)
      for i in range(len(result_shapes))
  ])


def _shape_with_default_layout(xla_shape: XlaShape) -> XlaShape:
  if not xla_shape.is_array():
    return xla_client.Shape.token_shape()
  else:
    return xla_client.Shape.array_shape(
        xla_shape.element_type(), xla_shape.dimensions(),
        tuple(range(xla_shape.rank() - 1, -1, -1)))
