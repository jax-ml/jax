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

from __future__ import annotations

import unittest
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax
from jax._src import core
from jax._src import config
from jax._src import test_util as jtu
from jax._src.util import safe_map, safe_zip
import jax.numpy as jnp

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


class WriterTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_writer_closed_over(self, jit):
    l = Log()
    def f(x):
      l.append(x)
      
    if jit:
      f = jax.jit(f)

    f(1)
    assert l.read() == [1]
    f(2)
    assert l.read() == [1, 2]

# TODO: generalize to other monoids (e.g. dict of lists)
class Log:
  def __init__(self):
    trace = core.trace_ctx.trace
    self._init_trace = trace

  def extend(self, vals):
    trace = core.trace_ctx.trace
    trace.loggers.setdefault(self, []).extend(vals)

  def append(self, val):
    self.extend([val]) 

  def read(self):
    trace = core.trace_ctx.trace
    if not trace == self._init_trace:
      raise Exception("Can only read a log in the same context as its creation")
    return trace.loggers.get(self, [])[:]

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

