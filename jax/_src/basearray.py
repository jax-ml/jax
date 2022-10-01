# Copyright 2022 The JAX Authors.
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

# Note that type annotations for this file are defined in basearray.pyi

import abc

class Array(abc.ABC):
  """Experimental Array base class for JAX

  `jax.Array` is meant as the future public interface for instance checks and
  type annotation of JAX array objects. JAX Array object types are currently in
  flux, and this class only fully supports the new `jax.experimental.Array`, which
  will soon replace the old-style {class}`DeviceArray`, {class}`ShardedDeviceArray`,
  {class}`GlobalDeviceArray`, etc.

  The compatibility is summarized in the following table:

  ================================  ======================  =========================
  object type                       ``isinstance`` support  type annotation support
  ================================  ======================  =========================
  {class}`DeviceArray`               ✅                      ❌
  {class}`ShardedDeviceArray`        ✅                      ❌
  {class}`GlobalDeviceArray`         ✅                      ❌
  {class}`~jax.core.Tracer`          ✅                      ✅
  {class}`~jax.experimental.Array`   ✅                      ✅
  ================================  ======================  =========================

  In other words, ``isinstance(x, jax.Array)`` will return True for any of these types,
  whereas annotations such as ``x : jax.Array`` will only type-check correctly for
  instances of {class}`~jax.core.Tracer` and {class}`jax.experimental.Array`, and not
  for the other soon-to-be-deprecated array types.
  """
  # Note: no abstract methods are defined in this base class; the associated pyi
  # file contains the type signature for static type checking.

  __slots__ = ['__weakref__']

  # at property must be defined because we overwrite its docstring in lax_numpy.py
  @property
  def at(self):
    raise NotImplementedError("property must be defined in subclasses")
