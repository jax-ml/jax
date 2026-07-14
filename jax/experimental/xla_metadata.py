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
or setting fusion boundaries).
"""

from __future__ import annotations

import contextlib

from jax._src.xla_metadata import (
    set_xla_metadata as set_xla_metadata,
    xla_metadata_call as xla_metadata_call,
)


@contextlib.contextmanager
def fuse_limit():
  """Context manager to set an XLA fusion boundary.

  Operations created within this context will be annotated with
  ``FUSE_LIMIT=True`` XLA metadata, acting as a boundary to prevent XLA from
  fusing operations across the limit.
  """

  with set_xla_metadata(FUSE_LIMIT=True):
    yield


def must_fuse_call(identifier: str):
  """Decorator to force XLA to fuse all operations within a function call.

  Annotates the wrapped function call with ``MUST_FUSE=identifier`` XLA
  metadata.


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


  Args:
    identifier: The identifier (numerical value) for the fusion group.

  Returns:
    A wrapped version of the function with the fusion metadata applied.
  """

  return xla_metadata_call(MUST_FUSE=identifier)
