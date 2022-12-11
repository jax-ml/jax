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

# Tests for lowering of JAX shapes and types into MLIR.

# RUN: %PYTHON %s | FileCheck %s

from absl import app

import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

from jax.tests.filecheck.jax_filecheck_helpers import print_ir

jax.config.update("jax_enable_x64", True)


def main(_):
  # CHECK-LABEL: TEST: bitwise_not bool[7]
  # CHECK: hlo.not
  # CHECK-SAME: tensor<7xi1>
  print_ir(np.empty([7], np.bool_))(lax.bitwise_not)

  # CHECK-LABEL: TEST: neg int8[]
  # CHECK: hlo.negate
  # CHECK-SAME: tensor<i8>
  print_ir(np.int8(0))(lax.neg)

  # CHECK-LABEL: TEST: neg int16[0]
  # CHECK: hlo.negate
  # CHECK-SAME: tensor<0xi16>
  print_ir(np.empty([0], np.int16))(lax.neg)

  # CHECK-LABEL: TEST: neg int32[2,3]
  # CHECK: hlo.negate
  # CHECK-SAME: tensor<2x3xi32>
  print_ir(np.empty([2, 3], np.int32))(lax.neg)

  # CHECK-LABEL: TEST: neg int64[2,3,4]
  # CHECK: hlo.negate
  # CHECK-SAME: tensor<2x3x4xi64>
  print_ir(np.empty([2,3,4], np.int64))(lax.neg)

  # CHECK-LABEL: TEST: add uint8[4,0,1] uint8[4,0,1]
  # CHECK: hlo.add
  # CHECK-SAME: tensor<4x0x1xui8>
  print_ir(np.empty([4,0,1], np.uint8), np.empty([4,0,1], np.uint8))(lax.add)

  # CHECK-LABEL: TEST: add uint16[] uint16[]
  # CHECK: hlo.add
  # CHECK-SAME: tensor<ui16>
  print_ir(np.uint16(0), np.uint16(0))(lax.add)

  # CHECK-LABEL: TEST: add uint32[] uint32[]
  # CHECK: hlo.add
  # CHECK-SAME: tensor<ui32>
  print_ir(np.uint32(0), np.uint32(0))(lax.add)

  # CHECK-LABEL: TEST: add uint64[] uint64[]
  # CHECK: hlo.add
  # CHECK-SAME: tensor<ui64>
  print_ir(np.uint64(0), np.uint64(0))(lax.add)

  # CHECK-LABEL: TEST: sin float16[]
  # CHECK: hlo.sine
  # CHECK-SAME: tensor<f16>
  print_ir(np.float16(0))(lax.sin)

  # CHECK-LABEL: TEST: sin bfloat16[]
  # CHECK: hlo.sine
  # CHECK-SAME: tensor<bf16>
  print_ir(jnp.bfloat16(0))(lax.sin)

  # CHECK-LABEL: TEST: sin float32[]
  # CHECK: hlo.sine
  # CHECK-SAME: tensor<f32>
  print_ir(np.float32(0))(lax.sin)

  # CHECK-LABEL: TEST: sin float64[]
  # CHECK: hlo.sine
  # CHECK-SAME: tensor<f64>
  print_ir(np.float64(0))(lax.sin)

  # CHECK-LABEL: TEST: cos complex64[]
  # CHECK: hlo.cosine
  # CHECK-SAME: tensor<complex<f32>>
  print_ir(np.complex64(0))(lax.cos)

  # CHECK-LABEL: TEST: cos complex128[]
  # CHECK: hlo.cosine
  # CHECK-SAME: tensor<complex<f64>>
  print_ir(np.complex128(0))(lax.cos)


if __name__ == "__main__":
  app.run(main)
