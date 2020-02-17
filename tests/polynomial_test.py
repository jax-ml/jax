# Copyright 2019 Google LLC
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


import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax import numpy as np
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


float_dtypes = [onp.float32, onp.float64]
# implementation casts to complex64.
complex_dtypes = [onp.complex64]
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = [onp.int32, onp.int64]
real_dtypes = float_dtypes + int_dtypes
all_dtypes = real_dtypes + complex_dtypes


class TestPolynomial(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "dtype={}_len={}_leading={}_trailing={}".format(
      jtu.format_shape_dtype_string((length,), dtype), length, leading, trailing),
      "dtype": dtype, "length": length, "rng_factory": rng_factory, "leading": leading, "trailing": trailing}
    for rng_factory in [jtu.rand_default]
    for dtype in all_dtypes
    for length in [3, 9, 10, 17]
    for leading in [1, 2, 3, 5, 7, 10]
    for trailing in [1, 2, 3, 5, 7, 10]))
  def testRoots(self, length, dtype, rng_factory, leading, trailing):
    rng = rng_factory()
    def args_maker():
      p = rng((length,), dtype)
      return (np.concatenate([np.zeros(leading, p.dtype), p, np.zeros(trailing, p.dtype)]),)

    # order may differ (np.sort doesn't deal with complex numbers)
    np_fn = lambda arg: onp.sort(np.roots(arg))
    onp_fn = lambda arg: onp.sort(onp.roots(arg))
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
