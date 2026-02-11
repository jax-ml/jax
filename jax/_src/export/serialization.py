# Copyright 2023 The JAX Authors.
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

# Serialization and deserialization of _export.Exported

from __future__ import annotations

import types
from collections.abc import Callable, Sequence
import itertools
from functools import partial
from typing import Any, TypeVar

try:
  import flatbuffers
except ImportError as e:
  raise ImportError(
      "Please install 'flatbuffers' in order to use Exported serialization"
      ) from e

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.export import serialization_generated as ser_flatbuf
from jax._src.export import _export
from jax._src.export import shape_poly
from jax._src.lib import xla_client
from jax._src import mesh
from jax._src import named_sharding
from jax._src import partition_spec
from jax._src import tree_util

import numpy as np

T = TypeVar("T")
SerT = TypeVar("SerT")

# The _SERIALIZATION_VERSION changes when we change the serialization schema
# even if the change is backwards compatible.
# Version 1, Nov 2023, first version.
# Version 2, Dec 16th, 2023, adds the f0 dtype.
# Version 3, October 16th, 2024, adds serialization for namedtuple and custom types
#   This version is backwards compatible with Version 2.
# Version 4, April 7th, 2025, adds serialization for PRNGs key types.
#   This version is backwards compatible with Version 2 and 3.
# Version 5, November 23rd, 2025, adds serialization for aval memory_space,
#   upgrade num_devices to a 32 bit value.
#   This version is backwards compatible with Version 2 to 4.
# Version 6, January 15th, 2026, adds serialization for sharding as
#   NamedSharding, including the abstract mesh, and the partition spec.
#   This version is backwards compatible with Version 2 to 5.
_SERIALIZATION_VERSION = 6

def serialize(exp: _export.Exported, vjp_order: int = 0) -> bytearray:
  """Serializes an Exported.

  Args:
    exp: the Exported to serialize.
    vjp_order: The maximum vjp order to include. E.g., the value 2 means that we
      serialize the primal functions and two orders of the `vjp` function. This
      should allow 2nd order reverse mode differentiation of the deserialized
      function. i.e., `jax.grad(jax.grad(f)).`
  """
  builder = flatbuffers.Builder(65536)
  exported = _serialize_exported(builder, exp, vjp_order)
  builder.Finish(exported)
  return builder.Output()


def deserialize(ser: bytearray) -> _export.Exported:
  """Deserializes an Exported."""
  exp = ser_flatbuf.Exported.GetRootAsExported(ser)
  return _deserialize_exported(exp)


def _serialize_exported(
    builder: flatbuffers.Builder, exp: _export.Exported, vjp_order: int
) -> int:
  # Serialize bottom-up
  fun_name = builder.CreateString(exp.fun_name)
  in_tree = _serialize_pytreedef(builder, exp.in_tree)
  in_avals = _serialize_array(builder, _serialize_aval, exp.in_avals)
  out_tree = _serialize_pytreedef(builder, exp.out_tree)
  out_avals = _serialize_array(builder, _serialize_aval, exp.out_avals)
  # TODO(necula): For 30 days after 1/15/2026 we must serialize the HLO
  # shardings. After that we can error out when serialized Exported with not
  # _has_named_shardings.
  in_shardings = _serialize_array(
      builder, partial(_serialize_sharding, has_named_sharding=exp._has_named_shardings),
      zip(exp._in_named_shardings, exp.in_shardings_hlo)  # type: ignore
  )
  out_shardings = _serialize_array(
      builder, partial(_serialize_sharding, has_named_sharding=exp._has_named_shardings),
      zip(exp._out_named_shardings, exp.out_shardings_hlo)  # type: ignore
  )
  ordered_effects = _serialize_array(
      builder, _serialize_effect, exp.ordered_effects
  )
  unordered_effects = _serialize_array(
      builder, _serialize_effect, exp.unordered_effects
  )
  disabled_safety_checks = _serialize_array(
      builder, _serialize_disabled_safety_check, exp.disabled_safety_checks
  )
  platforms = _serialize_array(
      builder, lambda b, p: b.CreateString(p), exp.platforms
  )
  mlir_module_serialized = builder.CreateByteVector(exp.mlir_module_serialized)
  module_kept_var_idx = builder.CreateNumpyVector(
      np.array(exp.module_kept_var_idx, dtype=np.uint16)
  )

  vjp = None
  if vjp_order > 0:
    if not exp.has_vjp():
      # TODO: add test
      raise ValueError(
          "serialization of an Exported that does not have vjps of high-enough "
          "order"
      )
    vjp = _serialize_exported(builder, exp.vjp(), vjp_order - 1)

  ser_flatbuf.ExportedStart(builder)
  # TODO(necula): we cannot really store the actual serialization_version
  # in the flatbuffer because prior to 11/25/2025 deserializers checked
  # if the version is 2 or 3. I have now removed that check, but for the
  # sake of old deserializers we can only store version 3. Starting
  # on January 2026 we can store the actual version.
  ser_flatbuf.ExportedAddSerializationVersion(builder, 3)
  ser_flatbuf.ExportedAddFunctionName(builder, fun_name)
  ser_flatbuf.ExportedAddInTree(builder, in_tree)
  ser_flatbuf.ExportedAddInAvals(builder, in_avals)
  ser_flatbuf.ExportedAddOutTree(builder, out_tree)
  ser_flatbuf.ExportedAddOutAvals(builder, out_avals)
  ser_flatbuf.ExportedAddNrDevices(builder, exp.nr_devices)
  ser_flatbuf.ExportedAddInShardings(builder, in_shardings)
  ser_flatbuf.ExportedAddOutShardings(builder, out_shardings)
  ser_flatbuf.ExportedAddPlatforms(builder, platforms)
  ser_flatbuf.ExportedAddOrderedEffects(builder, ordered_effects)
  ser_flatbuf.ExportedAddUnorderedEffects(builder, unordered_effects)
  ser_flatbuf.ExportedAddDisabledChecks(builder, disabled_safety_checks)
  ser_flatbuf.ExportedAddMlirModuleSerialized(builder, mlir_module_serialized)
  ser_flatbuf.ExportedAddCallingConventionVersion(
      builder, exp.calling_convention_version
  )
  ser_flatbuf.ExportedAddModuleKeptVarIdx(builder, module_kept_var_idx)
  ser_flatbuf.ExportedAddUsesGlobalConstants(
      builder, exp.uses_global_constants
  )
  if vjp is not None:
    ser_flatbuf.ExportedAddVjp(builder, vjp)
  return ser_flatbuf.ExportedEnd(builder)


def _serialize_array(
    builder: flatbuffers.Builder,
    serialize_one: Callable[[flatbuffers.Builder, T], int],
    elements: Sequence[T],
) -> int:
  element_offsets = [serialize_one(builder, e) for e in elements]
  ser_flatbuf.PyTreeDefStartChildrenVector(builder, len(element_offsets))
  for sc in reversed(element_offsets):
    builder.PrependUOffsetTRelative(sc)
  return builder.EndVector()


def _deserialize_exported(exp: ser_flatbuf.Exported) -> _export.Exported:
  fun_name = exp.FunctionName().decode("utf-8")
  in_tree = tree_util.tree_structure(
      _deserialize_pytreedef_to_pytree(exp.InTree())
  )
  scope = shape_poly.SymbolicScope(())  # TODO(necula): serialize the constraints
  out_tree = tree_util.tree_structure(
      _deserialize_pytreedef_to_pytree(exp.OutTree())
  )

  # TODO(necula): remove the fallback to NrDevicesShort and mark
  # the field "deprecated" once we abandon the old
  # serialization format (6 months after 11/24/2025).
  nr_devices = exp.NrDevices() or exp.NrDevicesShort()
  in_shardings = _deserialize_tuple(
      exp.InShardingsLength, exp.InShardings, _deserialize_sharding
  )
  out_shardings = _deserialize_tuple(
      exp.OutShardingsLength, exp.OutShardings, _deserialize_sharding
  )
  # has_named_sharding will be True for all exports after 1/15/2026
  has_named_shardings = not any(isinstance(s, _export.HloSharding)
                                for s in itertools.chain(in_shardings, out_shardings))
  if has_named_shardings:
    in_avals = tuple(
      _deserialize_aval(exp.InAvals(i), scope=scope, sharding=in_shardings[i])  # type: ignore
      for i in range(exp.InAvalsLength())
    )
    out_avals = tuple(
      _deserialize_aval(exp.OutAvals(i), scope=scope, sharding=out_shardings[i])  # type: ignore
      for i in range(exp.OutAvalsLength())
    )
    in_shardings_hlo = tuple(_export.named_to_hlo_sharding(s, aval)  # type: ignore
                             for s, aval in zip(in_shardings, in_avals))
    out_shardings_hlo = tuple(_export.named_to_hlo_sharding(s, aval)  # type: ignore
                             for s, aval in zip(out_shardings, out_avals))
  else:
    in_avals = _deserialize_tuple(exp.InAvalsLength, exp.InAvals,
                                  partial(_deserialize_aval, scope=scope, sharding=None))
    out_avals = _deserialize_tuple(exp.OutAvalsLength, exp.OutAvals,
                                   partial(_deserialize_aval, scope=scope, sharding=None))
    in_shardings_hlo, in_shardings = in_shardings, (None,) * len(in_shardings)  # type: ignore
    out_shardings_hlo, out_shardings = out_shardings, (None,) * len(out_shardings)  # type: ignore
  platforms = _deserialize_tuple(
      exp.PlatformsLength,
      exp.Platforms,
      lambda v: v.decode("utf-8"),
  )
  ordered_effects = _deserialize_tuple(
      exp.OrderedEffectsLength, exp.OrderedEffects, _deserialize_effect
  )
  unordered_effects = _deserialize_tuple(
      exp.UnorderedEffectsLength, exp.UnorderedEffects, _deserialize_effect
  )
  disabled_safety_checks = _deserialize_tuple(
      exp.DisabledChecksLength,
      exp.DisabledChecks,
      _deserialize_disabled_safety_check,
  )

  mlir_module_serialized = exp.MlirModuleSerializedAsNumpy().tobytes()
  calling_convention_version = exp.CallingConventionVersion()
  module_kept_var_idx = tuple(exp.ModuleKeptVarIdxAsNumpy().tolist())
  uses_global_constants = exp.UsesGlobalConstants()

  _get_vjp = None
  if vjp := exp.Vjp():
    _get_vjp = lambda _: _deserialize_exported(vjp)

  return _export.Exported(
      fun_name=fun_name,
      in_tree=in_tree,
      in_avals=in_avals,
      out_tree=out_tree,
      out_avals=out_avals,
      nr_devices=nr_devices,
      in_shardings_hlo=in_shardings_hlo,
      out_shardings_hlo=out_shardings_hlo,
      _has_named_shardings=has_named_shardings,
      _in_named_shardings=in_shardings,  # type: ignore
      _out_named_shardings=out_shardings,  # type: ignore
      platforms=platforms,
      ordered_effects=ordered_effects,
      unordered_effects=unordered_effects,
      disabled_safety_checks=disabled_safety_checks,
      mlir_module_serialized=mlir_module_serialized,
      calling_convention_version=calling_convention_version,
      module_kept_var_idx=module_kept_var_idx,
      uses_global_constants=uses_global_constants,
      _get_vjp=_get_vjp,
  )


def _deserialize_tuple(
    get_len: Callable[[], int],
    get_elem: Callable[[int], SerT],
    deserialize_one: Callable[[SerT], T],
) -> tuple[T, ...]:
  return tuple(deserialize_one(get_elem(i)) for i in range(get_len()))


def _serialize_pytreedef(
    builder: flatbuffers.Builder, p: tree_util.PyTreeDef
) -> int:
  node_data = p.node_data()
  children = p.children()

  children_vector_offset = None
  children_names_vector_offset = None
  if children:
    children_vector_offset = _serialize_array(
        builder, _serialize_pytreedef, children
    )
  custom_name = None
  custom_auxdata = None
  node_type = node_data and node_data[0]

  if node_data is None:  # leaf
    kind = ser_flatbuf.PyTreeDefKind.leaf
  elif node_type is types.NoneType:
    kind = ser_flatbuf.PyTreeDefKind.none
  elif node_type is tuple:
    kind = ser_flatbuf.PyTreeDefKind.tuple
  elif node_type is list:
    kind = ser_flatbuf.PyTreeDefKind.list
  elif node_type is dict:
    kind = ser_flatbuf.PyTreeDefKind.dict
    assert len(node_data[1]) == len(children)
    def serialize_key(builder, k):
      if not isinstance(k, str):
        raise TypeError(
            "Serialization is supported only for dictionaries with string keys."
            f" Found key {k} of type {type(k)}.")
      return builder.CreateString(k)
    children_names_vector_offset = _serialize_array(
        builder, serialize_key, node_data[1]
    )
  elif node_type in _export.serialization_registry:
    kind = ser_flatbuf.PyTreeDefKind.custom
    serialized_name, serialize_auxdata = _export.serialization_registry[node_type]
    custom_name = builder.CreateString(serialized_name)
    serialized_auxdata = serialize_auxdata(node_data[1])
    if not isinstance(serialized_auxdata, (bytes, bytearray)):
      raise ValueError(
          "The custom serialization function for `node_type` must "
          f"return a `bytes` object. It returned a {type(serialized_auxdata)}.")
    custom_auxdata = builder.CreateByteVector(serialized_auxdata)
  else:
    raise ValueError(
        "Cannot serialize PyTreeDef containing an "
        f"unregistered type `{node_type}`. "
        "Use `export.register_pytree_node_serialization` or "
        "`export.register_namedtuple_serialization`.")

  ser_flatbuf.PyTreeDefStart(builder)
  ser_flatbuf.PyTreeDefAddKind(builder, kind)
  if children_vector_offset:
    ser_flatbuf.PyTreeDefAddChildren(builder, children_vector_offset)
  if children_names_vector_offset:
    ser_flatbuf.PyTreeDefAddChildrenNames(builder, children_names_vector_offset)
  if custom_name is not None:
    ser_flatbuf.PyTreeDefAddCustomName(builder, custom_name)
  if custom_auxdata is not None:
    ser_flatbuf.PyTreeDefAddCustomAuxdata(builder, custom_auxdata)
  return ser_flatbuf.PyTreeDefEnd(builder)


def _deserialize_pytreedef_to_pytree(p: ser_flatbuf.PyTreeDef):
  # We construct a PyTree and later we'll flatten it to get the PyTreeDef.
  # TODO: is there a more direct way to construct a PyTreeDef?
  kind = p.Kind()
  nr_children = p.ChildrenLength()
  children = [
      _deserialize_pytreedef_to_pytree(p.Children(i))
      for i in range(nr_children)
  ]
  if kind == ser_flatbuf.PyTreeDefKind.leaf:
    return 0.0
  elif kind == ser_flatbuf.PyTreeDefKind.none:
    return None
  elif kind == ser_flatbuf.PyTreeDefKind.tuple:
    return tuple(children)
  elif kind == ser_flatbuf.PyTreeDefKind.list:
    return list(children)
  elif kind == ser_flatbuf.PyTreeDefKind.dict:
    assert p.ChildrenNamesLength() == nr_children
    keys = [p.ChildrenNames(i).decode("utf-8") for i in range(nr_children)]
    return dict(zip(keys, children))
  elif kind == ser_flatbuf.PyTreeDefKind.custom:
    serialized_name = p.CustomName().decode("utf-8")
    if serialized_name not in _export.deserialization_registry:
      raise ValueError(
          "Cannot deserialize a PyTreeDef containing an "
          f"unregistered type `{serialized_name}`. "
          "Use `export.register_pytree_node_serialization` or "
          "`export.register_namedtuple_serialization`.")
    nodetype, deserialize_auxdata, from_iter = _export.deserialization_registry[serialized_name]
    auxdata = deserialize_auxdata(p.CustomAuxdataAsNumpy().tobytes())
    return from_iter(auxdata, children)
  else:
    raise ValueError(
        f"Cannot deserialize PyTreeDef with unknown kind: {kind}")


_dtype_to_dtype_kind = {
    np.dtype("bool"): ser_flatbuf.DType.bool,
    np.dtype("int8"): ser_flatbuf.DType.i8,
    np.dtype("int16"): ser_flatbuf.DType.i16,
    np.dtype("int32"): ser_flatbuf.DType.i32,
    np.dtype("int64"): ser_flatbuf.DType.i64,
    np.dtype("uint8"): ser_flatbuf.DType.ui8,
    np.dtype("uint16"): ser_flatbuf.DType.ui16,
    np.dtype("uint32"): ser_flatbuf.DType.ui32,
    np.dtype("uint64"): ser_flatbuf.DType.ui64,
    dtypes.float0: ser_flatbuf.DType.f0,
    np.dtype("float16"): ser_flatbuf.DType.f16,
    np.dtype("float32"): ser_flatbuf.DType.f32,
    np.dtype("float64"): ser_flatbuf.DType.f64,
    np.dtype("complex64"): ser_flatbuf.DType.c64,
    np.dtype("complex128"): ser_flatbuf.DType.c128,
    dtypes._bfloat16_dtype: ser_flatbuf.DType.bf16,
    dtypes._int4_dtype: ser_flatbuf.DType.i4,
    dtypes._uint4_dtype: ser_flatbuf.DType.ui4,
    dtypes._float8_e4m3b11fnuz_dtype: ser_flatbuf.DType.f8_e4m3b11fnuz,
    dtypes._float8_e4m3fn_dtype: ser_flatbuf.DType.f8_e4m3fn,
    dtypes._float8_e4m3fnuz_dtype: ser_flatbuf.DType.f8_e4m3fnuz,
    dtypes._float8_e5m2_dtype: ser_flatbuf.DType.f8_e5m2,
    dtypes._float8_e5m2fnuz_dtype: ser_flatbuf.DType.f8_e5m2fnuz,
    dtypes._float8_e3m4_dtype: ser_flatbuf.DType.f8_e3m4,
    dtypes._float8_e4m3_dtype: ser_flatbuf.DType.f8_e4m3,
    dtypes._float8_e8m0fnu_dtype: ser_flatbuf.DType.f8_e8m0fnu,
    dtypes._float4_e2m1fn_dtype: ser_flatbuf.DType.f4_e2m1fn,
}

_dtype_kind_to_dtype = {
    kind: dtype for dtype, kind in _dtype_to_dtype_kind.items()
}


def register_dtype_kind(dtype: Any, kind: int):
  _dtype_to_dtype_kind[dtype] = kind
  _dtype_kind_to_dtype[kind] = dtype


_memory_space_to_enum = {
    core.MemorySpace.Device: ser_flatbuf.MemorySpace.Device,
    core.MemorySpace.Host: ser_flatbuf.MemorySpace.Host,
    core.MemorySpace.Any: ser_flatbuf.MemorySpace.Any,
}
_memory_space_from_enum = {v: k for k, v in _memory_space_to_enum.items()}


_axis_type_to_enum = {
    core.AxisType.Auto: ser_flatbuf.AxisType.Auto,
    core.AxisType.Explicit: ser_flatbuf.AxisType.Explicit,
    core.AxisType.Manual: ser_flatbuf.AxisType.Manual,
}
_axis_type_from_enum = {v: k for k, v in _axis_type_to_enum.items()}


def _serialize_abstract_mesh(builder: flatbuffers.Builder,
                             mesh: mesh.AbstractMesh) -> int:
  ser_flatbuf.AbstractMeshStartAxisSizesVector(builder, len(mesh.axis_sizes))
  for axis_size in reversed(mesh.axis_sizes):
    builder.PrependUint32(axis_size)
  axis_sizes = builder.EndVector()

  axis_names = _serialize_array(builder,
                                lambda builder, an: builder.CreateString(an),
                                mesh.axis_names)

  assert mesh.axis_types is not None, mesh
  ser_flatbuf.AbstractMeshStartAxisTypesVector(builder, len(mesh.axis_types))
  for axis_type in reversed(mesh.axis_types):
    builder.PrependByte(_axis_type_to_enum[axis_type])
  axis_types = builder.EndVector()

  ser_flatbuf.AbstractMeshStart(builder)
  ser_flatbuf.AbstractMeshAddAxisSizes(builder, axis_sizes)
  ser_flatbuf.AbstractMeshAddAxisNames(builder, axis_names)
  ser_flatbuf.AbstractMeshAddAxisTypes(builder, axis_types)
  return ser_flatbuf.AbstractMeshEnd(builder)


def _deserialize_abstract_mesh(ser_mesh: ser_flatbuf.AbstractMesh
) -> mesh.AbstractMesh:
  axis_sizes = tuple(ser_mesh.AxisSizes(i)
                     for i in range(ser_mesh.AxisSizesLength()))
  axis_names = tuple(ser_mesh.AxisNames(i).decode("utf-8")
                     for i in range(ser_mesh.AxisNamesLength()))
  axis_types = tuple(_axis_type_from_enum[ser_mesh.AxisTypes(i)]
                     for i in range(ser_mesh.AxisTypesLength()))
  return mesh.AbstractMesh(axis_sizes, axis_names, axis_types)


def _serialize_partition_spec_one_axis(builder: flatbuffers.Builder,
                                       spec: str | tuple[str, ...] | None) -> int:
  if spec is None:
    axes = ()
  else:
    axes = (spec,) if isinstance(spec, str) else spec  # type: ignore

  axes_offset = _serialize_array(builder,
                                 lambda builder, ps: builder.CreateString(ps),
                                 axes)
  ser_flatbuf.PartitionSpecOneAxisStart(builder)
  ser_flatbuf.PartitionSpecOneAxisAddAxes(builder, axes_offset)
  return ser_flatbuf.PartitionSpecOneAxisEnd(builder)


def _deserialize_partition_spec_one_axis(
    spec: ser_flatbuf.PartitionSpecOneAxis) -> str | tuple[str, ...] | None:
  axes = tuple(spec.Axes(i).decode("utf-8") for i in range(spec.AxesLength()))
  if not axes:
    return None
  else:
    return axes[0] if len(axes) == 1 else axes


def _serialize_partition_spec(builder: flatbuffers.Builder,
                              spec: partition_spec.PartitionSpec) -> int:
  partitions = _serialize_array(builder, _serialize_partition_spec_one_axis,
                                spec._partitions)
  reduced = _serialize_array(builder,  # type: ignore
                             lambda builder, ps: builder.CreateString(ps),
                             spec.reduced)
  unreduced = _serialize_array(builder,  # type: ignore
                               lambda builder, ps: builder.CreateString(ps),
                               spec.unreduced)

  ser_flatbuf.PartitionSpecStart(builder)
  ser_flatbuf.PartitionSpecAddPartitions(builder, partitions)
  ser_flatbuf.PartitionSpecAddReduced(builder, reduced)
  ser_flatbuf.PartitionSpecAddUnreduced(builder, unreduced)
  return ser_flatbuf.PartitionSpecEnd(builder)


def _deserialize_partition_spec(spec: ser_flatbuf.PartitionSpec
                                ) -> partition_spec.PartitionSpec:
  partitions = tuple(_deserialize_partition_spec_one_axis(spec.Partitions(i))
                     for i in range(spec.PartitionsLength()))
  reduced = frozenset(spec.Reduced(i).decode("utf-8")
                      for i in range(spec.ReducedLength()))
  unreduced = frozenset(spec.Unreduced(i).decode("utf-8")
                        for i in range(spec.UnreducedLength()))
  return partition_spec.PartitionSpec(*partitions,
                                      reduced=reduced,
                                      unreduced=unreduced)


def _serialize_named_sharding(
    builder: flatbuffers.Builder, sharding: named_sharding.NamedSharding
) -> int:
  mesh_offset = _serialize_abstract_mesh(builder, sharding.mesh.abstract_mesh)
  spec_offset = _serialize_partition_spec(builder, sharding.spec)
  memory_kind = builder.CreateString(sharding.memory_kind) if sharding.memory_kind is not None else 0

  ser_flatbuf.NamedShardingStart(builder)
  ser_flatbuf.NamedShardingAddMesh(builder, mesh_offset)
  ser_flatbuf.NamedShardingAddSpec(builder, spec_offset)
  if memory_kind != 0:
    ser_flatbuf.NamedShardingAddMemoryKind(builder, memory_kind)
  return ser_flatbuf.NamedShardingEnd(builder)


def _deserialize_named_sharding(
    s: ser_flatbuf.NamedSharding
) -> named_sharding.NamedSharding:
  amesh = _deserialize_abstract_mesh(s.Mesh())
  spec = _deserialize_partition_spec(s.Spec())
  memory_kind = s.MemoryKind().decode("utf-8") if s.MemoryKind() is not None else None
  return named_sharding.NamedSharding(amesh, spec, memory_kind=memory_kind)


def _serialize_aval(
    builder: flatbuffers.Builder, aval: core.ShapedArray
) -> int:
  aval_kind = ser_flatbuf.AbstractValueKind.shapedArray
  shape_offsets = [builder.CreateString(str(d)) for d in aval.shape]
  ser_flatbuf.AbstractValueStartShapeVector(builder, len(aval.shape))
  for d in reversed(shape_offsets):
    builder.PrependUOffsetTRelative(d)
  shape_vector_offset = builder.EndVector()

  ser_flatbuf.AbstractValueStart(builder)
  ser_flatbuf.AbstractValueAddKind(builder, aval_kind)
  ser_flatbuf.AbstractValueAddShape(builder, shape_vector_offset)
  ser_flatbuf.AbstractValueAddDtype(builder, _dtype_to_dtype_kind[aval.dtype])
  ser_flatbuf.AbstractValueAddMemorySpace(builder, _memory_space_to_enum[aval.memory_space])
  return ser_flatbuf.AbstractValueEnd(builder)


def _deserialize_aval(aval: ser_flatbuf.AbstractValue, *,
                      scope: shape_poly.SymbolicScope,
                      sharding: named_sharding.NamedSharding | None,
                      ) -> core.ShapedArray:
  dtype = _dtype_kind_to_dtype[aval.Dtype()]
  shape = shape_poly.symbolic_shape(
      ",".join(
            aval.Shape(i).decode("utf-8") for i in range(aval.ShapeLength())
        ),
        scope=scope
    )
  if (ser_mem_space := aval.MemorySpace()):
    mem_space = _memory_space_from_enum[ser_mem_space]
  else:
    mem_space = core.MemorySpace.Device

  aval = core.ShapedArray(shape, dtype, memory_space=mem_space)
  return core.update_aval_with_sharding(aval, sharding)


def _serialize_sharding(
    builder: flatbuffers.Builder, s: tuple[_export.NamedSharding | None, _export.HloSharding | None],
    has_named_sharding: bool,
) -> int:
  named_s, hlo_s = s
  is_unspecified = (named_s is None) if has_named_sharding else (hlo_s is None)
  named_sharding = None
  hlo_sharding = None
  if is_unspecified:
    kind = ser_flatbuf.ShardingKind.unspecified
  else:
    # TODO(necula): We must use the hlo_sharding kind for at least 30 days after
    # 1/15/2026 because old deserializers check the kind and abort if they
    # do not recognize it. After that date, we can stop using the kind.
    kind = ser_flatbuf.ShardingKind.hlo_sharding

  if has_named_sharding and named_s is not None:
    named_sharding = _serialize_named_sharding(builder, named_s)

  # TODO(necula): We must serialize the hlo_sharding for at least 30 days after
  # 1/15/2026 because old deserializers can only deserialize this sharding.
  if hlo_s is not None:
    hlo_sharding = builder.CreateByteVector(hlo_s.to_proto().SerializeToString())

  ser_flatbuf.ShardingStart(builder)
  ser_flatbuf.ShardingAddKind(builder, kind)
  if hlo_sharding is not None:
    ser_flatbuf.ShardingAddHloShardingProto(builder, hlo_sharding)
  if named_sharding is not None:
    ser_flatbuf.ShardingAddNamedSharding(builder, named_sharding)
  return ser_flatbuf.ShardingEnd(builder)


def _deserialize_sharding(s: ser_flatbuf.Sharding) -> _export.HloSharding | named_sharding.NamedSharding | None:
  if (named_sharding_off := s.NamedSharding()) is not None:
    # After 1/15/26 all exports will have named shardings (or None)
    return _deserialize_named_sharding(named_sharding_off)

  # TODO(necula): We must keep reading the HloSharding for 6 months after 1/15/2026.
  if not s.HloShardingProtoIsNone():
    proto = xla_client.OpSharding()
    proto.ParseFromString(s.HloShardingProtoAsNumpy().tobytes())
    return xla_client.HloSharding.from_proto(proto)

  return None  # Unspecified sharding


def _serialize_effect(builder: flatbuffers.Builder, eff: core.Effect) -> int:
  try:
    eff_replica = eff.__class__()
  except Exception:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary constructor to be serializable"
    )
  try:
    hash_eff = hash(eff)
    hash_eff_replica = hash(eff_replica)
  except Exception:
    raise NotImplementedError(
        f"Effect {eff} must be hashable to be serializable"
    )
  if eff != eff_replica or hash_eff != hash_eff_replica:
    raise NotImplementedError(
      f"Effect {eff} must have a nullary class constructor that produces an "
      "equal effect object."
    )
  effect_type_name = str(eff.__class__)
  effect_type_name_offset = builder.CreateString(effect_type_name)
  ser_flatbuf.EffectStart(builder)
  ser_flatbuf.EffectAddTypeName(builder, effect_type_name_offset)
  return ser_flatbuf.ExportedEnd(builder)


def _deserialize_effect(eff: ser_flatbuf.Effect) -> core.Effect:
  effect_type_name = eff.TypeName().decode("utf-8")
  for existing_effect_type in effects.lowerable_effects._effect_types:
    if str(existing_effect_type) == effect_type_name:
      try:
        return existing_effect_type()
      except:
        # TODO: add test
        raise NotImplementedError(
            f"deserializing effect {effect_type_name} that does not have a "
            "nullary class constructor"
        )

  raise NotImplementedError(
      f"cannot deserialize effect type {effect_type_name}"
  )


def _serialize_disabled_safety_check(
    builder: flatbuffers.Builder, check: _export.DisabledSafetyCheck
) -> int:
  custom_call_target_str = check.is_custom_call()
  custom_call_target = None
  if custom_call_target_str is not None:
    kind = ser_flatbuf.DisabledSafetyCheckKind.custom_call
    custom_call_target = builder.CreateString(custom_call_target_str)
  elif check == _export.DisabledSafetyCheck.platform():
    kind = ser_flatbuf.DisabledSafetyCheckKind.platform
  else:
    raise NotImplementedError(f"serializing DisabledSafetyCheck: {check}")

  ser_flatbuf.DisabledSafetyCheckStart(builder)
  ser_flatbuf.DisabledSafetyCheckAddKind(builder, kind)
  if custom_call_target is not None:
    ser_flatbuf.DisabledSafetyCheckAddCustomCallTarget(
        builder, custom_call_target
    )
  return ser_flatbuf.DisabledSafetyCheckEnd(builder)


def _deserialize_disabled_safety_check(
    sc: ser_flatbuf.DisabledSafetyCheck,
) -> _export.DisabledSafetyCheck:
  kind = sc.Kind()
  if kind == ser_flatbuf.DisabledSafetyCheckKind.custom_call:
    return _export.DisabledSafetyCheck.custom_call(
        sc.CustomCallTarget().decode("utf-8")
    )
  if kind == ser_flatbuf.DisabledSafetyCheckKind.platform:
    return _export.DisabledSafetyCheck.platform()
  if kind == ser_flatbuf.DisabledSafetyCheckKind.shape_assertions:
    # shape_assertions has been deprecated in June 2024 (turned into a no-op),
    # and removed in November 2024. We deserialize it to a DisabledSafetyCheck
    # that has no effect.
    # TODO(necula): remove this after June 2025, when we should not have any
    # more serialized artifacts with shape_assertions.
    return _export.DisabledSafetyCheck.custom_call("no op")
  raise ValueError(f"Cannot deserialize DisabledSafetyCheck with unknown kind: {kind}")
