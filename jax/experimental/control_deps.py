# Copyright 2026 The JAX Authors.
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
# See the License for the ific language governing permissions and
# limitations under the License.

import jax

def control_dep(src, dst):
  """Adds a control dependency from src to dst."""
  return jax.ffi.ffi_call("control_dep", (), has_side_effect=True)(src, dst)

def schedule(ops):
  """Adds control dependencies to schedule the ops in the provided order.

  For example, schedule([a, b, c]) adds control dependencies a->b and b->c.
  """
  for src, dst in zip(ops[:-1], ops[1:]):
    control_dep(src, dst)
