# Copyright 2021 The JAX Authors.
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

# flake8: noqa: F401
import jax._src.lib.mlir.dialects.builtin as builtin
import jax._src.lib.mlir.dialects.chlo as chlo
import jax._src.lib.mlir.dialects.mhlo as mhlo
import jax._src.lib.mlir.dialects.func as func
import jax._src.lib.mlir.dialects.ml_program as ml_program
import jax._src.lib.mlir.dialects.sparse_tensor as sparse_tensor

from jax._src import lib
# TODO(sharadmv): remove guard when minimum jaxlib version is bumped
if lib.jaxlib_version >= (0, 4, 15):
  import jax._src.lib.mlir.dialects.arith as arith
  import jax._src.lib.mlir.dialects.math as math
  import jax._src.lib.mlir.dialects.memref as memref
  import jax._src.lib.mlir.dialects.scf as scf
  import jax._src.lib.mlir.dialects.vector as vector
del lib

# Alias that is set up to abstract away the transition from MHLO to StableHLO.
import jax._src.lib.mlir.dialects.stablehlo as hlo
