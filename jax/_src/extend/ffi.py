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

from collections.abc import Callable, Mapping, Sequence
import ctypes
import functools
import os
from typing import Any

import numpy as np

from jax._src import core
from jax._src import deprecations
from jax._src import dispatch
from jax._src import effects
from jax._src import util
from jax._src.callback import _check_shape_dtype, callback_batching_rule
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.layout import DeviceLocalLayout
from jax._src.lib import jaxlib
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.typing import (Array, ArrayLike, DeprecatedArg, DuckTypedArray,
                             Shape)

# TODO(dfm): Remove after 6 months or less because there aren't any offical
# compatibility guarantees for jax.extend (see JEP 15856)
# Added Oct 13, 2024
deprecations.register("jax-ffi-call-args")

map, unsafe_map = util.safe_map, map
FfiLayoutOptions = Sequence[int] | DeviceLocalLayout | None


def register_ffi_target(
    name: str,
    fn: Any,
    platform: str = "cpu",
    api_version: int = 1,
    **kwargs: Any,
) -> None:
  """Registers a foreign function target.

  Args:
    name: the name of the target.
    fn: a ``PyCapsule`` object containing the function pointer, or a ``dict``
      where the keys are FFI stage names (e.g. `"execute"`) and the values are
      ``PyCapsule`` objects continaing a pointer to the handler for that stage.
    platform: the target platform.
    api_version: the XLA custom call API version to use. Supported versions are:
      1 (default) for the typed FFI or 0 for the earlier "custom call" API.
    kwargs: any extra keyword arguments are passed directly to
      :func:`~jaxlib.xla_client.register_custom_call_target` for more advanced
      use cases.
  """
  return xla_client.register_custom_call_target(name, fn, platform, api_version,
                                                **kwargs)


def pycapsule(funcptr):
  """Wrap a ctypes function pointer in a PyCapsule.

  The primary use of this function, and the reason why it lives with in the
  ``jax.extend.ffi`` submodule, is to wrap function calls from external
  compiled libraries to be registered as XLA custom calls.

  Example usage::

    import ctypes
    import jax
    from jax.lib import xla_client

    libfoo = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
        name="bar",
        fn=jax.extend.ffi.pycapsule(libfoo.bar),
        platform=PLATFORM,
        api_version=API_VERSION
    )

  Args:
    funcptr: A function pointer loaded from a dynamic library using ``ctypes``.

  Returns:
    An opaque ``PyCapsule`` object wrapping ``funcptr``.
  """
  destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
  builder = ctypes.pythonapi.PyCapsule_New
  builder.restype = ctypes.py_object
  builder.argtypes = (ctypes.c_void_p, ctypes.c_char_p, destructor)
  return builder(funcptr, None, destructor(0))


def include_dir() -> str:
  """Get the path to the directory containing header files bundled with jaxlib"""
  jaxlib_dir = os.path.dirname(os.path.abspath(jaxlib.__file__))
  return os.path.join(jaxlib_dir, "include")


def _aval_shape(aval: core.AbstractValue) -> Shape:
  return () if aval is core.abstract_token else aval.shape  # pytype: disable=attribute-error


def _convert_layout(aval: core.AbstractValue,
                    layout: FfiLayoutOptions = None) -> Sequence[int]:
  """Convert a layout to the minor-to-major order used by the custom call API."""
  if layout is None:
    return list(reversed(range(len(_aval_shape(aval)))))
  elif isinstance(layout, DeviceLocalLayout):
    if layout._tiling is not None:
      raise ValueError("The FFI does not support layouts with tiling")
    return layout.major_to_minor[::-1]
  else:
    return layout


def ffi_lowering(
    call_target_name: str,
    *,
    operand_layouts: Sequence[FfiLayoutOptions] | None = None,
    result_layouts: Sequence[FfiLayoutOptions] | None = None,
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
      backend_config or {}, **{k: mlir.ir_attribute(v) for k, v in params.items()})
    if "result_types" not in kwargs:
      kwargs["result_types"] = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    if operand_layouts is None:
      kwargs["operand_layouts"] = map(_convert_layout, ctx.avals_in)
    else:
      kwargs["operand_layouts"] = [
          _convert_layout(*args) for args in zip(ctx.avals_in, operand_layouts)]
    if result_layouts is None:
      kwargs["result_layouts"] = map(_convert_layout, ctx.avals_out)
    else:
      kwargs["result_layouts"] = [
          _convert_layout(*args) for args in zip(ctx.avals_out, result_layouts)]
    if "result_shapes" not in kwargs and not all(
        core.is_constant_shape(_aval_shape(aval)) for aval in ctx.avals_out):
      kwargs["result_shapes"] = [
          mlir.shape_tensor(mlir.eval_dynamic_shape_as_ivals(ctx, _aval_shape(aval)))
          for aval in ctx.avals_out]

    return mlir.custom_call(call_target_name, operands=operands, **kwargs).results  # type: ignore

  return _lowering


ResultMetadata = DuckTypedArray | core.AbstractToken


def _result_avals(results: Sequence[ResultMetadata]) -> tuple[core.AbstractValue, ...]:
  avals: list[core.AbstractValue] = []
  for result in results:
    if isinstance(result, core.AbstractToken):
      avals.append(result)
    else:
      _check_shape_dtype(result)
      avals.append(core.ShapedArray(result.shape, result.dtype))
  return tuple(avals)


def ffi_call(
    target_name: str,
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],
    *deprecated_args: ArrayLike,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    vectorized: bool | DeprecatedArg = DeprecatedArg(),
    **deprecated_kwargs: Any,
) -> Callable[..., Array | Sequence[Array]] | Array | Sequence[Array]:
  """Call a foreign function interface (FFI) target.

  See the :ref:`ffi-tutorial` tutorial for more information.

  Like :func:`~jax.pure_callback`, the behavior of ``ffi_call`` under
  :func:`~jax.vmap` depends on the value of ``vmap_method``. See the
  :func:`~jax.pure_callback` documenation for more details about the allowed
  values and examples of their behavior.

  The current default behavior is to use ``vmap_method="sequential"`` when
  not specified, but this behavior is deprecated, and in the future, the
  default will be to raise a ``NotImplementedError`` unless ``vmap_method`` is
  explicitly specified.

  Args:
    target_name: the name of the XLA FFI custom call target that was registered
      using :func:`~jaxlib.xla_client.register_custom_call_target`.
    result_shape_dtypes: an object, or sequence of objects, with ``shape`` and
      ``dtype`` attributes which are expected to match the shape and dtype of
      the custom call output or outputs. :class:`~jax.ShapeDtypeStruct` is often
      used to define the elements of ``result_shape_dtypes``.
      ``jax.core.abstract_token`` may be used to represent a token-typed output.
    has_side_effect: boolean specifying whether the custom call has side
      effects. When ``True``, the FFI call will be executed even when the
      outputs are not used.
    vmap_method: string specifying how the FFI call transforms under
      :func:`~jax.vmap` as described above.

  Returns:
    A function that can be called with the input arrays as positional arguments
    to execute the FFI handler. Any keyword arguments are passed as named
    attributes to the FFI handler using XLA's FFI interface.
  """
  if not isinstance(vectorized, DeprecatedArg) and not vectorized is None:
    deprecations.warn(
        "jax-callback-vectorized",
        "The vectorized argument of ffi_call is deprecated and setting "
        "it will soon raise an error. To avoid an error in the future, and to "
        "suppress this warning, please use the vmap_method argument instead.",
        stacklevel=2)
    if vmap_method is not None:
      raise ValueError(
          "the vectorized and vmap_method arguments of ffi_call cannot "
          "be used together. Please use the vmap_method argument.")
    vmap_method = "legacy_vectorized" if vectorized else "sequential"
  allowed_vmap_methods = ["sequential", "expand_dims", "broadcast_all",
                          "legacy_vectorized", None]
  if vmap_method not in allowed_vmap_methods:
    raise ValueError(
        f"vmap_method must be on of the allowed methods {allowed_vmap_methods}, "
        f"but got: {vmap_method}")

  if isinstance(result_shape_dtypes, Sequence):
    multiple_results = True
    result_avals = _result_avals(result_shape_dtypes)
  else:
    multiple_results = False
    result_avals = _result_avals((result_shape_dtypes,))

  def wrapped(*args: ArrayLike, **kwargs: Any):
    results = ffi_call_p.bind(
        *args,
        result_avals=result_avals,
        vectorized=vectorized,
        vmap_method=vmap_method,
        target_name=target_name,
        has_side_effect=has_side_effect,
        attributes=_wrap_kwargs_hashable(kwargs),
    )
    if multiple_results:
      return results
    else:
      return results[0]

  if deprecated_args or deprecated_kwargs:
    deprecations.warn(
        "jax-ffi-call-args",
        "Calling ffi_call directly with input arguments is deprecated. "
        "Instead, ffi_call should be used to construct a callable, which can "
        "then be called with the appropriate inputs. For example,\n"
        "  ffi_call('target_name', output_type, x, argument=5)\n"
        "should be replaced with\n"
        "  ffi_call('target_name', output_type)(x, argument=5)",
        stacklevel=2)
    return wrapped(*deprecated_args, **deprecated_kwargs)
  else:
    return wrapped


# ffi_call must support some small non-hashable input arguments, like np.arrays
# and dicts, to support calling FFI targets with array inputs or user defined
# structs. Since these arguments will eventually be embedded in the HLO as
# dense attributes, we assume that they are small and hash by making an
# immutable copy and hashing by value.
def _wrap_kwargs_hashable(kwargs: dict[str, Any]) -> Sequence[tuple[str, Any]]:
  hashable_kwargs: list[tuple[str, Any]] = []
  for k, v in sorted(kwargs.items()):
    if isinstance(v, np.ndarray):
      hashable_kwargs.append((k, HashableArray(v)))
    elif isinstance(v, dict):
      hashable_kwargs.append((k, HashableDict(v)))
    else:
      try:
        hash(v)
      except TypeError as e:
        raise TypeError(
            f"Non-hashable keyword argument to ffi_call {k}: {v}") from e
      else:
        hashable_kwargs.append((k, v))
  return tuple(hashable_kwargs)


def _unwrap_kwargs_hashable(kwargs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  unwrapped_kwargs: dict[str, Any] = {}
  for k, v in kwargs:
    if isinstance(v, HashableArray):
      unwrapped_kwargs[k] = v.val
    elif isinstance(v, HashableDict):
      unwrapped_kwargs[k] = dict(v.val)
    else:
      unwrapped_kwargs[k] = v
  return unwrapped_kwargs


class HashableArray:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, np.ndarray)
    self.val = np.copy(val)
    self.val.setflags(write=False)

  def __repr__(self):
    return f"HashableArray({self.val})"

  def __hash__(self):
    return hash((self.val.shape, self.val.dtype, self.val.tobytes()))

  def __eq__(self, other):
    return isinstance(other, HashableArray) and np.array_equal(self.val, other.val)


class HashableDict:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, dict)
    self.val = tuple(sorted(val.items()))

  def __repr__(self):
    return f"HashableDict({dict(self.val)})"

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return isinstance(other, HashableDict) and self.val == other.val


class FfiEffect(effects.Effect):
  def __str__(self):
    return "FFI"


_FfiEffect = FfiEffect()
effects.lowerable_effects.add_type(FfiEffect)
effects.control_flow_allowed_effects.add_type(FfiEffect)


def ffi_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    target_name: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    attributes: Sequence[tuple[str, Any]],
):
  del avals_in, target_name, vectorized, vmap_method, attributes
  effects = {_FfiEffect} if has_side_effect else core.no_effects
  return result_avals, effects


def ffi_call_jvp(*args, target_name, **kwargs):
  del args, kwargs
  raise ValueError(
      f"The FFI call to `{target_name}` cannot be differentiated. "
      "You can use `jax.custom_jvp` or `jax.custom_jvp` to add support.")


def ffi_call_transpose(*args, target_name, **kwargs):
  del args, kwargs
  raise ValueError(
      f"The FFI call to `{target_name}` cannot be differentiated. "
      "You can use `jax.custom_jvp` or `jax.custom_jvp` to add support.")


def ffi_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    result_avals: tuple[core.AbstractValue, ...],
    target_name: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    attributes: Sequence[tuple[str, Any]],
) -> Sequence[ir.Value]:
  del result_avals, vectorized, vmap_method
  rule = ffi_lowering(target_name, has_side_effect=has_side_effect)
  return rule(ctx, *operands, **_unwrap_kwargs_hashable(attributes))


ffi_call_p = core.Primitive("ffi_call")
ffi_call_p.multiple_results = True
dispatch.simple_impl(ffi_call_p)
ffi_call_p.def_effectful_abstract_eval(ffi_call_abstract_eval)
ad.primitive_jvps[ffi_call_p] = ffi_call_jvp
ad.primitive_transposes[ffi_call_p] = ffi_call_transpose
batching.primitive_batchers[ffi_call_p] = functools.partial(
    callback_batching_rule, ffi_call_p)
mlir.register_lowering(ffi_call_p, ffi_call_lowering)
