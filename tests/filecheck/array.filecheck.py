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

# Tests for lowering of array origami ops into MLIR.

# RUN: %PYTHON %s | FileCheck %s

from absl import app
from functools import partial

import jax
from jax import lax
import numpy as np

from jax._src.lax import lax as lax_internal

from jax.tests.filecheck.jax_filecheck_helpers import print_ir

jax.config.update("jax_enable_x64", True)


def main(_):
  # CHECK-LABEL: TEST: concatenate bool[2,7] bool[2,5]
  # CHECK: hlo.concatenate
  # CHECK-SAME: tensor<2x12xi1>
  print_ir([np.empty([2, 7], np.bool_), np.empty([2, 5], np.bool_)])(
      partial(lax.concatenate, dimension=1))

  # CHECK-LABEL: TEST: broadcast_in_dim bool[2,7]
  # CHECK: hlo.broadcast_in_dim
  # CHECK-SAME: tensor<3x2x5x7x2xi1>
  print_ir(np.empty([2, 7], np.bool_))(
      partial(lax.broadcast_in_dim, shape=(3, 2, 5, 7, 2),
              broadcast_dimensions=(1, 3)))

  # CHECK-LABEL: TEST: iota
  # CHECK: hlo.iota
  # CHECK-SAME: tensor<10xf32>
  print_ir()(partial(lax.iota, dtype=np.float32, size=10))

  # CHECK-LABEL: TEST: pad int32[2,7]
  # CHECK: hlo.pad
  # CHECK-SAME: tensor<11x52xi32>
  print_ir(np.empty([2, 7], np.int32))(
      partial(lax.pad, padding_value=np.int32(7),
              padding_config=((2, 3, 4), (4, 5, 6))))

  # CHECK-LABEL: TEST: _reduce_sum int32[2,3,7]
  # CHECK: hlo.reduce
  # CHECK: hlo.add
  # CHECK: tensor<3xi32>
  print_ir(np.empty([2, 3, 7], np.int32))(
      partial(lax_internal._reduce_sum, axes=(0, 2)))

  # CHECK-LABEL: TEST: reshape int32[2,3,7]
  # CHECK: hlo.reshape
  # CHECK-SAME: tensor<42xi32>
  print_ir(np.empty([2, 3, 7], np.int32))(
      partial(lax.reshape, new_sizes=(42,)))

  # CHECK-LABEL: TEST: rev int32[2,7]
  # CHECK: hlo.rev
  # CHECK-SAME: tensor<2x7xi32>
  print_ir(np.empty([2, 7], np.int32))(
      partial(lax.rev, dimensions=(0, 1)))

  # CHECK-LABEL: TEST: select bool[2,7] int32[2,7] int32[2,7]
  # CHECK: hlo.select
  # CHECK-SAME: tensor<2x7xi1>, tensor<2x7xi32>
  print_ir(np.empty([2, 7], np.bool_), np.empty([2, 7], np.int32),
           np.empty([2, 7], np.int32))(lax.select)

  # CHECK-LABEL: TEST: sort int32[2,7]
  # CHECK: hlo.sort
  # CHECK: tensor<2x7xi32>
  print_ir(np.empty([2, 7], np.int32))(lax.sort)

  # CHECK-LABEL: TEST: squeeze int32[2,1,7]
  # CHECK: hlo.reshape
  # CHECK-SAME: tensor<2x7xi32>
  print_ir(np.empty([2, 1, 7], np.int32))(
      partial(lax.squeeze, dimensions=(1,)))

  # CHECK-LABEL: TEST: top_k int32[2,7]
  # CHECK: chlo.top_k
  # CHECK: tensor<2x7xi32>
  print_ir(np.empty([2, 7], np.int32))(partial(lax.top_k, k=7))

  # CHECK-LABEL: TEST: transpose int32[2,7]
  # CHECK: hlo.transpose
  # CHECK-SAME: tensor<7x2xi32>
  print_ir(np.empty([2, 7], np.int32))(
      partial(lax.transpose, permutation=(1, 0)))

if __name__ == "__main__":
  app.run(main)
