# Copyright 2025 The JAX Authors.
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

"""Multihost tests for pgle."""

import functools
import math
import os
import tempfile

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np


class PgleTestMultiHost(jt_multiprocess.MultiProcessTest):

  def get_fdo_profiles(self, dump_dir):
    jit_f_fdo_profiles = [
        x
        for x in os.listdir(dump_dir)
        if 'jit_f' in x and x.endswith('.fdo_profile')
    ]
    return jit_f_fdo_profiles

  @parameterized.parameters(True, False)
  def testAutoPGLE(self, use_compilation_cache: bool):
    mesh = jtu.create_mesh((jax.device_count(),), ('x',))

    its = 500

    with tempfile.TemporaryDirectory() as dump_dir:

      @functools.partial(
          jax.jit,
          in_shardings=NamedSharding(mesh, PartitionSpec('x')),
          out_shardings=NamedSharding(mesh, PartitionSpec('x')),
          compiler_options={
              'xla_gpu_enable_latency_hiding_scheduler': 'True',
              # TODO(patrios): Remove this flag once b/376647494 is fixed.
              'xla_gpu_graph_min_graph_size': '100000',
              'xla_dump_to': dump_dir,
              'xla_gpu_experimental_dump_fdo_profiles': 'True',
          },
      )
      def f(x):
        agg = x
        for _ in range(its):
          agg = agg @ x
        return agg

      shape = (16, 16)
      x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

      num_runs = 2
      with (
          config.pgle_profiling_runs(num_runs),
          config.enable_pgle(True),
          config.enable_compilation_cache(use_compilation_cache),
          config.raise_persistent_cache_errors(True),
          config.raise_persistent_cache_errors(True),
          config.persistent_cache_min_entry_size_bytes(0),
          config.persistent_cache_min_compile_time_secs(0),
      ):
        for _ in range(num_runs):
          f(x)

        # There should be 3 fdo profiles: before optimization, after
        # SPMD-partitioning, and after optimization.
        fdo_profiles_before_pgle = self.get_fdo_profiles(dump_dir)
        self.assertLen(fdo_profiles_before_pgle, 3)
        self.assertEqual(
            os.path.getsize(
                os.path.join(dump_dir, fdo_profiles_before_pgle[0])
            ),
            0,
        )

        # Should recompile with the FDO profile.
        f(x)

        # Expect 3 additional non-empty fdo profiles.
        fdo_profiles_after_pgle = self.get_fdo_profiles(dump_dir)
        self.assertLen(fdo_profiles_after_pgle, 6)
        for fdo_profile in fdo_profiles_after_pgle:
          if fdo_profile not in fdo_profiles_before_pgle:
            self.assertGreater(
                os.path.getsize(os.path.join(dump_dir, fdo_profile)), 0
            )


if __name__ == '__main__':
  jt_multiprocess.main()
