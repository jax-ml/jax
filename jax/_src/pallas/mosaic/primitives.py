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

"""Module for Pallas:TPU-specific JAX primitives and functions."""
from __future__ import annotations

import dataclasses
import enum
from typing import Any

import jax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import pretty_printer as pp
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.pallas import core as pl_core
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.state.types import Transform
from jax._src.typing import DTypeLike
import jax.numpy as jnp

Slice = indexing.Slice

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

repeat_p = jax_core.Primitive('repeat')

def repeat(x, repeats, axis):
  return repeat_p.bind(x, repeats=repeats, axis=axis)

@repeat_p.def_abstract_eval
def _repeat_abstract_eval(x, *, repeats, axis):
  shape = list(x.shape)
  shape[axis] *= repeats
  return jax_core.ShapedArray(shape, x.dtype)


def _repeat_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, repeats, axis):
  def _repeat(x):
    return jnp.repeat(x, repeats, axis)
  return mlir.lower_fun(_repeat, multiple_results=False)(ctx, x)
mlir.register_lowering(repeat_p, _repeat_lowering_rule)

bitcast_p = jax_core.Primitive("bitcast")


def bitcast(x, ty: DTypeLike):
  ty = dtypes.canonicalize_dtype(ty)
  if len(x.shape) < 2:
    raise ValueError("Not implemented: bitcast 1D")
  src_bitwidth = pallas_utils.dtype_bitwidth(x.dtype)
  dst_bitwidth = pallas_utils.dtype_bitwidth(ty)
  if x.shape[-2] * src_bitwidth % dst_bitwidth:
    raise ValueError(
        "Not implemented: the 2nd minor dim can not be perfectly packed or"
        " unpacked"
    )
  return bitcast_p.bind(x, ty=ty)


@bitcast_p.def_abstract_eval
def _bitcast_abstract_eval(x, *, ty):
  shape = list(x.shape)
  src_bitwidth = pallas_utils.dtype_bitwidth(x.dtype)
  dst_bitwidth = pallas_utils.dtype_bitwidth(ty)
  shape[-2] = shape[-2] * src_bitwidth // dst_bitwidth
  return jax_core.ShapedArray(shape, ty)


def _bitcast_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, ty):
  def _bitcast(x):
    src_bitwidth = pallas_utils.dtype_bitwidth(x.dtype)
    dst_bitwidth = pallas_utils.dtype_bitwidth(ty)
    if src_bitwidth < dst_bitwidth:
      *leading, m, n = x.shape
      packing = dst_bitwidth // src_bitwidth
      x = x.reshape(*leading, m // packing, packing, n)
      x = jnp.swapaxes(x, -1, -2)
      return jax.lax.bitcast_convert_type(x, ty)
    if src_bitwidth > dst_bitwidth:
      y = jax.lax.bitcast_convert_type(x, ty)
      *leading, m, n, packing = y.shape
      return jnp.swapaxes(y, -1, -2).reshape(*leading, m * packing, n)
    return jax.lax.bitcast_convert_type(x, ty)

  return mlir.lower_fun(_bitcast, multiple_results=False)(ctx, x)


mlir.register_lowering(bitcast_p, _bitcast_lowering_rule)

roll_p = jax_core.Primitive("roll")


def roll(
    x,
    shift,
    axis: int,
    *,
    stride: int | None = None,
    stride_axis: int | None = None,
):
  if isinstance(shift, int) and shift < 0:
    raise ValueError("shift must be non-negative.")
  if axis < 0 or axis >= len(x.shape):
    raise ValueError("axis is out of range.")
  if (stride is None) != (stride_axis is None):
    raise ValueError("stride and stride_axis must be both specified or not.")
  if stride is not None and stride_axis is not None:
    if stride < 0:
      raise ValueError("stride must be non-negative.")
    if stride_axis < 0 or stride_axis >= len(x.shape):
      raise ValueError("stride_axis is out of range")
    if axis == stride_axis:
      raise ValueError("expected axis and stride_axis are different.")
  return roll_p.bind(
      x, shift, axis=axis, stride=stride, stride_axis=stride_axis
  )


@roll_p.def_abstract_eval
def _roll_abstract_eval(x, shift, **_):
  del shift
  return x


def _roll_lowering_rule(
    ctx: mlir.LoweringRuleContext, x, shift, *, axis, stride, stride_axis
):
  def _roll(x, shift):
    if stride is None:
      return jnp.roll(x, shift, axis)
    outputs = [
        jnp.roll(xs, shift + i * stride, axis)
        for i, xs in enumerate(jnp.split(x, x.shape[stride_axis], stride_axis))
    ]
    return jnp.concatenate(outputs, stride_axis)

  return mlir.lower_fun(_roll, multiple_results=False)(ctx, x, shift)


mlir.register_lowering(roll_p, _roll_lowering_rule)

get_barrier_semaphore_p = jax_core.Primitive('get_barrier_semaphore')

@get_barrier_semaphore_p.def_abstract_eval
def _get_barrier_semaphore_abstract_eval():
  return pl_core.AbstractMemoryRef(
      jax_core.ShapedArray((), tpu_core.BarrierSemaphoreTy()),
      tpu_core.TPUMemorySpace.SEMAPHORE,
  )

def get_barrier_semaphore():
  """Returns a barrier semaphore.

  This function returns a barrier semaphore based on the collective_id of the
  current pallas kernel.

  It's very important that the semaphore is wait-ed back down to 0, or else the
  semaphores will become corrupted.

  It's also very important that the collective_id is different for each pallas
  kernel with communication. E.g. if you have two pallas kernels, one that syncs
  across the X axis of the device mesh and the second that syncs across the Y
  axis, they must have different collective_ids.
  However it is legal for two kernels that perform the same synchronization
  pattern (e.g. only communicating with neighbours on the same mesh axis)
  to share a collective_id. However, if in doubt, prefer not sharing
  collective_ids, as doing so incorrectly can lead to silent data corruption or
  crashes.
  Note that re-using the same collective_id doesn't guarantee that the same
  semaphore is provided by XLA.
  """
  return get_barrier_semaphore_p.bind()

delay_p = jax_core.Primitive("delay")
delay_p.multiple_results = True


@delay_p.def_abstract_eval
def _delay_abstract_eval(nanos):
  del nanos
  return []


def delay(nanos):
  """Delays vector execution for the given number of nanosconds."""
  delay_p.bind(nanos)


# RNG Ops
prng_seed_p = jax_core.Primitive("prng_seed")
prng_seed_p.multiple_results = True

@prng_seed_p.def_abstract_eval
def _(*_):
  return []


def prng_seed(*seeds: int | jax.Array) -> None:
  """Sets the seed for PRNG.

  Args:
    seeds: One or more integer seeds for setting the PRNG seed. If
      more than one seed is passed in, the seed material will be
      mixed before setting the internal PRNG state.
  """
  prng_seed_p.bind(*seeds)

prng_random_bits_p = jax_core.Primitive(
    'prng_random_bits')

@prng_random_bits_p.def_abstract_eval
def _(*, shape):
  return jax_core.ShapedArray(shape, jnp.dtype("int32"))

def prng_random_bits(shape):
  return prng_random_bits_p.bind(shape=shape)
