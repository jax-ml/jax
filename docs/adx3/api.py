# ---
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

from core import *
from util import *
from lax import *

def jax_type_of(x: LoosePyVal) -> JaxType:
  return canonicalize_pyval(x).ty

def jit(f): assert False

def value_and_grad(f, *args):
  args = map(canonicalize_pyval, args)
  jaxpr = trace_to_jaxpr(f, [arg.ty for arg in args])
  assert jaxpr.ty.result_type == JaxArrayType((), np.float64)
  (value, linearized) = linearize_jaxpr(jaxpr, args)
  gradient = transpose_linear_jaxpr(linearized, canonicalize_pyval(1.0))
  return value, gradient
