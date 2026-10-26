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

"""Experimental XLA metadata APIs.

These context managers, decorators, and function wrappers allow annotating JAX
operations with XLA metadata to attach arbitrary attributes or explicitly
control XLA's fusion decisions (e.g., forcing operations to be fused together
or setting fusion boundaries) and scheduling decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from typing import Any

from jax._src.xla_metadata import (
    set_xla_metadata as set_xla_metadata,
    xla_metadata_call2 as xla_metadata_call2,
    xla_metadata_call as xla_metadata_call,
)


def _must_fuse_kwargs(
    identifier: Any,
    buffering_level: int | None = None,
    kernel_window_bounds: Sequence[int] | None = None,
    output_window_bounds: Sequence[int] | None = None,
    scoped_vmem_limit_kib: int | None = None,
    integer_config: int | None = None,
) -> dict[str, Any]:
  if integer_config is not None and (
      buffering_level is not None
      or kernel_window_bounds is not None
      or output_window_bounds is not None
  ):
    raise ValueError(
        "integer_config cannot be set if buffering_level,"
        " kernel_window_bounds, or output_window_bounds is specified, and vice"
        " versa."
    )
  kwargs = {"MUST_FUSE": str(identifier)}
  if buffering_level is not None:
    kwargs["buffering_level"] = str(buffering_level)
  if kernel_window_bounds is not None:
    kwargs["kernel_window_bounds"] = ",".join(map(str, kernel_window_bounds))
  if output_window_bounds is not None:
    kwargs["output_window_bounds"] = ",".join(map(str, output_window_bounds))
  if scoped_vmem_limit_kib is not None:
    kwargs["scoped_vmem_limit_kib"] = str(scoped_vmem_limit_kib)
  if integer_config is not None:
    kwargs["integer"] = str(integer_config)
  return kwargs


@contextlib.contextmanager
def fuse_limit():
  """Context manager to set an XLA fusion boundary.

  Operations created within this context will be annotated with
  ``FUSE_LIMIT=True`` XLA metadata, acting as a boundary to prevent XLA from
  fusing operations across the limit.
  """

  with set_xla_metadata(FUSE_LIMIT=True):
    yield


# Frontend attributes of XLA scheduling groups.
_SCHEDULING_GROUP_ID_ATTR = "_scheduling_group_id"
_SCHEDULING_GROUP_DEEP_PIPELINE_ATTR = "_scheduling_group_deep_pipeline"


@contextlib.contextmanager
def scheduling_group(group_id: int, *, deep_pipeline: bool = False):
  """Context manager to put operations in an XLA scheduling group.

  Operations created within this context are annotated with
  ``_scheduling_group_id=group_id`` XLA metadata. The latency hiding scheduler
  schedules the operations of a group together, so that the asynchronous
  operations of the group (e.g. collectives) overlap with its computations.
  Operations that share the same ``group_id`` belong to the same group, even if
  they are created in different ``with`` blocks (for example, in different
  functions).

  With ``deep_pipeline=True``, the group is also in deep pipelining mode: a
  collective and the computation that consumes its result can be merged by the
  compiler into a single deep-pipelined operation, where the computation starts
  on each chunk of data as soon as it arrives. Unlike regular scheduling groups,
  such a group may contain operations that depend on each other. The roles of
  the operations (which one produces data and which one consumes it) are
  inferred from the data flow. On TPU, this currently pipelines a SparseCore
  ``ragged_all_to_all`` with the TensorCore ``ragged_dot`` (or ``dot_general``)
  consuming its result, without enabling SparseCore-TensorCore pipelining
  globally. Operations the compiler cannot pipeline are compiled as usual.

  Example:

    .. code-block:: python

      from jax.experimental.xla_metadata import scheduling_group

      with scheduling_group(7, deep_pipeline=True):
        x = jax.lax.ragged_all_to_all(x, out, *offsets_and_sizes, axis_name=ax)
      ...
      with scheduling_group(7, deep_pipeline=True):
        y = jax.lax.ragged_dot(x, w, group_sizes)

  Args:
    group_id: The non-negative integer identifier of the scheduling group.
    deep_pipeline: Whether the group is in deep pipelining mode. The mode
      applies to the whole group as soon as one of its operations sets it.

  Raises:
    ValueError: If ``group_id`` is not a non-negative integer.
  """
  if (
      isinstance(group_id, bool)
      or not isinstance(group_id, int)
      or group_id < 0
  ):
    raise ValueError(
        f"group_id must be a non-negative integer, got {group_id!r}."
    )
  metadata = {
      _SCHEDULING_GROUP_ID_ATTR: str(group_id),
      # `None` clears a mode set by an enclosing `scheduling_group`, so that it
      # does not leak into this group.
      _SCHEDULING_GROUP_DEEP_PIPELINE_ATTR: True if deep_pipeline else None,
  }
  with set_xla_metadata(**metadata):
    yield


def must_fuse_call(
    identifier: Any,
    buffering_level: int | None = None,
    kernel_window_bounds: Sequence[int] | None = None,
    output_window_bounds: Sequence[int] | None = None,
    scoped_vmem_limit_kib: int | None = None,
    integer_config: int | None = None,
):
  """Decorator to force XLA to fuse all operations within a function call.

  Annotates the wrapped function call with ``MUST_FUSE=identifier`` XLA
  metadata, along with optional compiler lowering configuration attributes for
  pipeline buffering, window bounds, scoped VMEM limits, or integer backend
  configs.


  Example:


    .. code-block:: python


      import jax
      import jax.numpy as jnp
      from jax.experimental.xla_metadata import must_fuse_call


      @jax.jit
      def f(x):
        y = jnp.sin(x)
        z = must_fuse_call('1')(lambda x: jnp.square(x).sum())(x)
        return y, z


    This results in the following ``must_fuse`` call in the before optimization
    HLO:


    .. code-block:: text


      %xla_metadata_call.2 (Arg_0.1: f32[128]) -> f32[] {
        %Arg_0.1 = f32[128]{0} parameter(0)
        %square.1 = f32[128]{0} multiply(%Arg_0.1, %Arg_0.1)
        %constant.1 = f32[] constant(0)
        ROOT %reduce_sum.7 = f32[] reduce(%square.1, %constant.1),
        dimensions={0},
        to_apply=%region_0.1
      }


      ENTRY main {
        ...
        %xla_metadata_call.1 = f32[] call(%x.1),
        to_apply=%xla_metadata_call.2,
        frontend_attributes={MUST_FUSE="1"}
        ...
      }


    After HLO optimization passes, all the operators within this call are
    guaranteed to be
    part of a single outermost fusion as seen below. Note that the instructions
    may form nested-fusion.


    .. code-block:: text


      %fused_computation (param_0.2: f32[128]) -> (f32[], f32[128]) {
        %param_0.2 = f32[128]{0:T(128)} parameter(0)
        %square.2 = f32[128]{0:T(128)} multiply(%param_0.2, %param_0.2),
        frontend_attributes={MUST_FUSE="1"}
        %constant.2 = f32[]{:T(128)} constant(0),
        frontend_attributes={MUST_FUSE="1"}
        %reduce_sum.1 = f32[]{:T(128)} reduce(%square.2, %constant.2),
        dimensions={0}, to_apply=%region_0.1,
        frontend_attributes={MUST_FUSE="1"}
        %sin.0 = f32[128]{0:T(128)} sine(%param_0.2)
        ROOT %tuple = (f32[]{:T(128)}, f32[128]{0:T(128)}) tuple(%reduce_sum.1,
        %sin.0)
      }


      ENTRY main {
        ...
        %multiply_reduce_fusion = (f32[]{:T(128)}, f32[128]{0:T(128)})
        fusion(%x.1), kind=kLoop, calls=%fused_computation,
        frontend_attributes={MUST_FUSE="1"}
        ...
      }


    Also note that ``must_fuse`` does not prevent other non-must-fuse
    instructions
    from fusing into the must-fuse fusion computation. However, not being able
    to
    form the must-fuse fusion condition is a compile time error.

    Optional lowering parameters can also be provided to configure the fused
    computation on TPU:


    .. code-block:: python


      @must_fuse_call(
          '2',
          buffering_level=3,
          kernel_window_bounds=(16, 24, 4),
          output_window_bounds=(16, 64, 8),
          scoped_vmem_limit_kib=8192,
      )
      def fused_matmul(x, y):
        return jnp.einsum('bmk,bkn->bmn', x, y)


  Args:
    identifier: The identifier (string or numerical value) for the fusion group.
    buffering_level: Optional integer specifying the pipeline buffering level
      (for example, 2 for double buffering or 3 for triple buffering) for the
      fused computation. Mutually exclusive with ``integer_config``.
    kernel_window_bounds: Optional sequence of integers specifying the kernel
      (RHS) window bounds for the fused computation. Mutually exclusive with
      ``integer_config``.
    output_window_bounds: Optional sequence of integers specifying the output
      window bounds for the fused computation. Mutually exclusive with
      ``integer_config``.
    scoped_vmem_limit_kib: Optional integer specifying the scoped VMEM limit in
      KiB allocated for the fused computation.
    integer_config: Optional integer specifying an integer backend configuration
      value. Mutually exclusive with ``buffering_level``,
      ``kernel_window_bounds``, and ``output_window_bounds``.

  Returns:
    A wrapped version of the function with the fusion metadata applied.

  Raises:
    ValueError: If ``integer_config`` is specified together with
      ``buffering_level``, ``kernel_window_bounds``, or
      ``output_window_bounds``.
  """

  kwargs = _must_fuse_kwargs(
      identifier,
      buffering_level,
      kernel_window_bounds,
      output_window_bounds,
      scoped_vmem_limit_kib,
      integer_config,
  )
  return xla_metadata_call(**kwargs)
