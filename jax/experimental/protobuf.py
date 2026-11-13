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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental APIs for reading protobuf files written by XLA.

XLA writes various protobuf files to disk, for example HLO dumps, profiles and
autotuning results. The ``protobuf`` package must be installed.

Example::

  from jax.experimental import protobuf

  hlo = protobuf.parse_file(
      "/tmp/dump/module_0001.jit_f.after_optimizations.hlo.pb", "xla.HloProto")
  print(hlo.hlo_module.name)
"""

from jax._src.protobuf import (
    load_pool as load_pool,
    parse as parse,
    parse_file as parse_file,
)
