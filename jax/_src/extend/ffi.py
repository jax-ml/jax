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

import os
import ctypes
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from jax._src import dtypes
from jax._src.interpreters import mlir
from jax._src.lib import jaxlib
from jax._src.lib.mlir import ir
from jax._src.typing import DimSize


def pycapsule(funcptr):
  """Construct a PyCapsule out of a ctypes function pointer.

  A typical use for this is registering custom call targets with XLA:

    import ctypes
    import jax
    from jax.lib import xla_client

    fooso = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
        name="bar",
        fn=jax.extend.ffi.pycapsule(fooso.bar),
        platform=PLATFORM,
        api_version=API_VERSION
    )
  """
  # Note (https://docs.python.org/3/library/ctypes.html):
  #
  #  ctypes.pythonapi
  #    An instance of PyDLL that exposes Python C API functions as attributes.
  #    Note that all these functions are assumed to return C int, which is of
  #    course not always the truth, so you have to assign the correct restype
  #    attribute to use these functions.
  #
  # Following this advice we annotate argument and return types of
  # PyCapsule_New before we call it, based on the example here:
  # https://stackoverflow.com/questions/65056619/converting-ctypes-c-void-p-to-pycapsule
  PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
  PyCapsule_New = ctypes.pythonapi.PyCapsule_New
  PyCapsule_New.restype = ctypes.py_object
  PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
  return PyCapsule_New(funcptr, None, PyCapsule_Destructor(0))


def include_dir() -> str:
  """Get the path to the directory containing header files bundled with jaxlib"""
  jaxlib_dir = os.path.dirname(os.path.abspath(jaxlib.__file__))
  return os.path.join(jaxlib_dir, "include")


def ffi_lowering(
    call_target_name: str,
    *,
    operand_layouts: Sequence[Sequence[DimSize]] | None = None,
    result_layouts: Sequence[Sequence[DimSize]] | None = None,
    backend_config: Mapping[str, ir.Attribute] | None = None,
    **lowering_args: Any
) -> mlir.LoweringRule:
  """Build a lowering rule for an foreign function interface (FFI) target.

  By default, this lowering rule can use the input and output abstract values to
  compute the input and output types and shapes for the custom call, assuming
  row-major layouts.

  If keyword arguments are passed to the lowering rule, these are treated as
  attributes, and added to `backend_config`.

  Args:
    call_target_name: The name of the custom call target.
    operand_layouts: A sequence of layouts (dimension orders) for each operand.
      By default, the operands are assumed to be row-major.
    result_layouts: A sequence of layouts (dimension orders) for each result.
      By default, the results are assumed to be row-major.
    backend_config: Configuration data for the custom call. Any keyword
      arguments passed to the lowering rule will added to this dictionary.
    lowering_args: Any other arguments to :func:`mlir.custom_call` will also be
      passed through if provided as extra arguments to this function.
  """

  def _lowering(
    ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> Sequence[ir.Value | Sequence[ir.Value]]:
    kwargs = dict(lowering_args)
    kwargs.setdefault("api_version", 4)
    kwargs["backend_config"] = dict(
      backend_config or {}, **{k: _ir_attribute(v) for k, v in params.items()})
    if "result_types" not in kwargs:
      kwargs["result_types"] = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    if operand_layouts is None:
      kwargs["operand_layouts"] = _default_layouts(aval.shape for aval in ctx.avals_in)  # pytype: disable=attribute-error
    if result_layouts is None:
      kwargs["result_layouts"] = _default_layouts(aval.shape for aval in ctx.avals_out)

    return mlir.custom_call(call_target_name, operands=operands, **kwargs).results  # type: ignore

  return _lowering


def _default_layouts(shapes: Iterable[Sequence[DimSize]]) -> list[list[DimSize]]:
  return [list(reversed(range(len(shape)))) for shape in shapes]


def _ir_attribute(obj: Any) -> ir.Attribute:
  # TODO(dfm): Similar functions exist in Pallas and Mosaic GPU. Perhaps these
  # could be consolidated into mlir or similar.
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  elif isinstance(obj, bool):
    return ir.BoolAttr.get(obj)
  elif isinstance(obj, int):
    return mlir.i64_attr(obj)
  elif isinstance(obj, float):
    return ir.FloatAttr.get_f64(obj)
  elif hasattr(obj, "dtype"):
    if not (dtypes.is_python_scalar(obj) or np.isscalar(obj)):
      raise TypeError("Only scalar attributes are supported")
    mlir_type = mlir.dtype_to_ir_type(obj.dtype)
    if isinstance(mlir_type, ir.IntegerType):
      return ir.IntegerAttr.get(mlir_type, obj)
    elif isinstance(mlir_type, ir.FloatType):
      return ir.FloatAttr.get(mlir_type, obj)
  raise TypeError(f"Unsupported attribute type: {type(obj)}")
