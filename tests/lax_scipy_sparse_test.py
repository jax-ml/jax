# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import lax
from jax import numpy as np

import unittest
import numpy as onp
from functools import partial
from jax import test_util as jtu
from jax.experimental import sparse
import scipy.sparse.linalg as osp_sparse


def high_precision_dot(a, b):
  return lax.dot(a, b, precision=lax.Precision.HIGHEST)

def build_and_solve(a, b):
  # intentionally non-linear in a and b
  matvec = partial(high_precision_dot, np.exp(a))
  return sparse.cg(matvec, np.cos(b))

rng = onp.random.RandomState(0)
a = rng.randn(2, 2)
b = rng.randn(2)
expected = osp_sparse.cg(np.exp(a), np.cos(b))
actual = build_and_solve(a, b)
self.assertAllClose(expected, actual, atol=1e-5, check_dtypes=True)
jtu.check_grads(build_and_solve, (a, b), atol=1e-5, order=2, rtol=2e-3)
