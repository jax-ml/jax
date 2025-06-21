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
"""An example demontrating the basic end-to-end use of the JAX FFI.

This example is exactly the same as the one in the `FFI tutorial
<https://docs.jax.dev/en/latest/ffi.html>`, so more details can be found
on that page. But, the high level summary is that we implement our custom
extension in ``rms_norm.cc``, then call it using ``jax.ffi.ffi_call`` in
this module. The behavior under autodiff is implemented using
``jax.custom_vjp``.
"""

from functools import partial

import numpy as np

import jax

from jax_ffi_example import _rms_norm

for name, target in _rms_norm.registrations().items():
  jax.ffi.register_ffi_target(name, target)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def rms_norm(x, eps=1e-5):
  # In this case, the output of our FFI function is just a single array with the
  # same shape and dtype as the input.
  out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

  # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
  # the attribute `eps`. Our FFI function expects this to have the C++ `float`
  # type (which corresponds to numpy's `float32` type), and it must be a
  # static parameter (i.e. not a JAX array).
  return jax.ffi.ffi_call(
    # The target name must be the same string as we used to register the target
    # above in `register_ffi_target`
    "rms_norm",
    out_type,
    vmap_method="broadcast_all",
  )(x, eps=np.float32(eps))


def rms_norm_fwd(x, eps=1e-5):
  y, res = jax.ffi.ffi_call(
    "rms_norm_fwd",
    (
      jax.ShapeDtypeStruct(x.shape, x.dtype),
      jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),
    ),
    vmap_method="broadcast_all",
  )(x, eps=np.float32(eps))
  return y, (res, x)


def rms_norm_bwd(eps, res, ct):
  del eps
  res, x = res
  assert res.shape == ct.shape[:-1]
  assert x.shape == ct.shape
  return (
    jax.ffi.ffi_call(
      "rms_norm_bwd",
      jax.ShapeDtypeStruct(ct.shape, ct.dtype),
      vmap_method="broadcast_all",
    )(res, x, ct),
  )


rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)
