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

from __future__ import annotations

from typing import Any, Callable

from jax import core
from jax.experimental.key_reuse import _forwarding
from jax.experimental.key_reuse import _simple
import numpy as np

# TODO(jakevdp) fix this
KeyReuseSignature = Any


def check_key_reuse(fun: Callable[..., Any], /, *args: Any,
                    use_forwarding: bool = True) -> KeyReuseSignature:
  """Function to statically check key reuse."""
  if use_forwarding:
    return _forwarding.check_key_reuse(fun, *args)
  else:
    return _simple.check_key_reuse(fun, *args)


def check_key_reuse_jaxpr(jaxpr: core.Jaxpr, *, use_forwarding: bool = True):
  """Check the jaxpr for key reuse."""
  get_jaxpr_type_signature(jaxpr, use_forwarding=use_forwarding)


def get_jaxpr_type_signature(
    jaxpr: core.Jaxpr, *,
    consumed_inputs: list[bool | np.ndarray] | None = None,
    use_forwarding: bool = True,
    ) -> KeyReuseSignature:
  """Parse the jaxpr to determine key reuse signature"""
  if use_forwarding:
    return _forwarding.get_jaxpr_type_signature(jaxpr, consumed_inputs)
  else:
    return _simple.get_jaxpr_type_signature(jaxpr, consumed_inputs)
