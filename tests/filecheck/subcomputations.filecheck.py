# Copyright 2021 Google LLC
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

# Tests for lowering of array origami ops into MHLO.

# RUN: %PYTHON %s | FileCheck %s

from absl import app

import jax
from jax import numpy as jnp
import numpy as np

from jax.tests.filecheck.jax_filecheck_helpers import print_ir

jax.config.update("jax_enable_mlir", True)
jax.config.update("jax_enable_x64", True)


def main(_):

  # The lowering of cumsum is annotated with @cache_lowering, which means we
  # should lower it as an out-of-line function once for any given shape.

  # CHECK-LABEL: TEST: cumsum_only_once int32[2,7] int32[2,7]
  # CHECK: func private @cumsum
  # CHECK-NOT: func private @cumsum
  @print_ir(np.empty([2, 7], np.int32), np.empty([2, 7], np.int32))
  def cumsum_only_once(x, y):
    return jnp.cumsum(x) + jnp.cumsum(y)

if __name__ == "__main__":
  app.run(main)
