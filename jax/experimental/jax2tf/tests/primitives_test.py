# Copyright 2020 Google LLC
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
"""Tests for JAX primitive coverage.

The bulk of the testing is done by `test_prim`, which is parameterized by
about 2000+ test harnesses. See `primitive_harness.py` docstring for a
description of test harnesses. That module contains also the definitions
of all the test harnesses, and a specification of which are only partially
implemented for JAX.

For each harness, we convert the JAX function to Tensorflow and then we run
it on the same inputs in "eager", "graph", or "compiled" mode and we check
that we get the same result as in JAX
(see `tf_test_util.ConvertAndCompare`).

Some harnesses need specific tolerances, or even custom equality assertions.
Also, for some harnesses we need to specify some data types that result
in Tensorflow errors (for some devices and compilation modes). These limitations
are captured as jax2tf_limitations.Jax2TfLimitation objects.

From the limitations objects, we generate a
[report](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).
The report has instructions for how to re-generate it.

If a harness run fails with error, and a limitation that matches the device
and data types is found,
the error is logged but does not abort the test. If a harness run succeeds
and there are matching limitations, the test emits a warning. If you want to
turn these warnings into errors, you'd have to uncomment an assertion
in `tf_test_util.ConvertAndCompare`.

IMPORTANT: If you need to customize the testing of a particular primitive
conversion, you must create a class method in jax2tf_limitations.jax2tf_limitations,
with the same name as the harness.group_name (typically the same as the
primitive name). That class method should return the list of Jax2TfLimitation
objects for the harness.
See `jax2tf_limitations.limitations_for_harness`. If a group name does
not need limitations, then it must be listed in the
`jax2tf_limitations.harness_groups_no_limitations`.

"""

import datetime
import os
from typing import Any, Dict, Tuple
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.config import config
from jax.experimental import jax2tf
from jax.interpreters import mlir
from jax.interpreters import xla

import numpy as np
import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation
from jax.experimental.jax2tf.tests import primitive_harness

DType = Any

REDUCE = (
    jnp.all,
    jnp.any,
    jnp.max,
    jnp.min,
    jnp.prod,
    jnp.sum,
)


class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  # This test runs for all primitive harnesses. For each primitive "xxx" the
  # test will be called "test_prim_xxx_..." and the custom parameters for
  # the test are defined in the class method "jax2tf_limitations.Jax2TfLimitation.xxx".
  # See more details in the comment at top of file and in Jax2TfLimitation class.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @primitive_harness.parameterized(
      primitive_harness.all_harnesses,
      include_jax_unimpl=False,
      #one_containing="custom_linear_solve_"
  )
  @jtu.ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*")
  def test_prim(self, harness: primitive_harness.Harness):
    limitations = Jax2TfLimitation.limitations_for_harness(harness)
    device = jtu.device_under_test()
    limitations = tuple(filter(lambda l: l.filter(device=device,
                                                  dtype=harness.dtype), limitations))
    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())
    enable_xla = harness.params.get("enable_xla", True)
    if config.jax2tf_default_experimental_native_lowering and not enable_xla:
      return
    associative_scan_reductions = harness.params.get("associative_scan_reductions", False)
    print(limitations, flush=True)
    try:
      with jax.jax2tf_associative_scan_reductions(associative_scan_reductions):
        self.ConvertAndCompare(func_jax, *args, limitations=limitations,
                               enable_xla=enable_xla)
    except Exception as e:
      if (config.jax2tf_default_experimental_native_lowering and
          "does not work with custom calls" in str(e)):
        logging.warning("Supressing error %s", e)
      else:
        raise e


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
