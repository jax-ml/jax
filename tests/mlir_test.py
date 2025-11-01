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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import numpy as np

jax.config.parse_flags_with_absl()


def _arr(value, dtype=None):
  return np.array(value, dtype=dtype)


def _ftens1(et):
  return f"dense<1.000000e+00> : tensor<{et}>"


def _itens1(et):
  return f"dense<1> : tensor<{et}>"


class MlirTest(jtu.JaxTestCase):

  @parameterized.parameters(
      # go/keep-sorted start
      (_arr(1, dtypes.int2), _itens1("i2")),
      (_arr(1, dtypes.int4), _itens1("i4")),
      (_arr(1, dtypes.uint2), _itens1("ui2")),
      (_arr(1, dtypes.uint4), _itens1("ui4")),
      (_arr(1, np.int16), _itens1("i16")),
      (_arr(1, np.int32), _itens1("i32")),
      (_arr(1, np.int64), _itens1("i64")),
      (_arr(1, np.int8), _itens1("i8")),
      (_arr(1, np.uint16), _itens1("ui16")),
      (_arr(1, np.uint32), _itens1("ui32")),
      (_arr(1, np.uint64), _itens1("ui64")),
      (_arr(1, np.uint8), _itens1("ui8")),
      (_arr(1.0, dtypes.bfloat16), _ftens1("bf16")),
      (_arr(1.0, dtypes.float4_e2m1fn), _ftens1("f4E2M1FN")),
      (_arr(1.0, dtypes.float8_e3m4), _ftens1("f8E3M4")),
      (_arr(1.0, dtypes.float8_e4m3), _ftens1("f8E4M3")),
      (_arr(1.0, dtypes.float8_e4m3b11fnuz), _ftens1("f8E4M3B11FNUZ")),
      (_arr(1.0, dtypes.float8_e4m3fn), _ftens1("f8E4M3FN")),
      (_arr(1.0, dtypes.float8_e4m3fnuz), _ftens1("f8E4M3FNUZ")),
      (_arr(1.0, dtypes.float8_e5m2), _ftens1("f8E5M2")),
      (_arr(1.0, dtypes.float8_e5m2fnuz), _ftens1("f8E5M2FNUZ")),
      (_arr(1.0, dtypes.float8_e8m0fnu), _ftens1("f8E8M0FNU")),
      (_arr(1.0, np.bool), "dense<true> : tensor<i1>"),
      (_arr(1.0, np.float16), _ftens1("f16")),
      (_arr(1.0, np.float32), _ftens1("f32")),
      (_arr(1.0, np.float64), _ftens1("f64")),
      (dtypes.bfloat16(1.0), "1.000000e+00 : bf16"),
      (np.bool(False), "false"),
      (np.bool(True), "true"),
      (np.float16(1.0), "1.000000e+00 : f16"),
      (np.float32(1.0), "1.000000e+00 : f32"),
      (np.float64(1.0), "1.000000e+00 : f64"),
      (np.int16(1), "1 : i16"),
      (np.int32(1), "1 : i32"),
      (np.int64(1), "1 : i64"),
      (np.int8(1), "1 : i8"),
      (np.uint16(1), "1 : ui16"),
      (np.uint32(1), "1 : ui32"),
      (np.uint64(1), "1 : ui64"),
      (np.uint8(1), "1 : ui8"),
      (np.zeros((), dtype=dtypes.float0), "dense<false> : tensor<i1>"),
      # go/keep-sorted end
  )
  def test_ir_attribute(self, value, expected):
    with jax_mlir.make_ir_context(), ir.Location.unknown():
      attribute = jax_mlir.ir_attribute(value)
    self.assertIsInstance(attribute, ir.Attribute)
    self.assertEqual(str(attribute), expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
