# Copyright 2022 Google LLC
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
import contextlib
import io
import unittest
from unittest import mock

from typing import Callable, Generator

from absl.testing import absltest
import jax
from jax.config import config
from jax._src import debugging
from jax._src import lib as jaxlib
from jax._src import test_util as jtu

config.parse_flags_with_absl()

debug_print = debugging.debug_print

@contextlib.contextmanager
def capture_stdout() -> Generator[Callable[[], str], None, None]:
  with mock.patch('sys.stdout', new_callable=io.StringIO) as fp:
    def _read() -> str:
      return fp.getvalue()
    yield _read

class DebugPrintTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("tpu", "gpu")
  def test_simple_debug_print_works_in_eager_mode(self):
    def f(x):
      debug_print('x: {}', x)
    with capture_stdout() as output:
      f(2)
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices("tpu", "gpu")
  def test_debug_print_works_with_named_format_strings(self):
    def f(x):
      debug_print('x: {x}', x=x)
    with capture_stdout() as output:
      f(2)
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices("tpu", "gpu")
  def test_multiple_debug_prints_should_print_multiple_values(self):
    def f(x):
      debug_print('x: {x}', x=x)
      debug_print('y: {y}', y=x + 1)
    with capture_stdout() as output:
      f(2)
    self.assertEqual(output(), "x: 2\ny: 3\n")

  @jtu.skip_on_devices("tpu", "gpu")
  def test_can_stage_out_debug_print(self):
    if jaxlib.version < (0, 3, 8):
      raise unittest.SkipTest(
          "`emit_python_callback` only supported in jaxlib >= 0.3.8")
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x)
    with capture_stdout() as output:
      f(2)
    self.assertEqual(output(), "x: 2\n")

  @jtu.skip_on_devices("tpu", "gpu")
  def test_can_stage_out_ordered_print(self):
    if jaxlib.version < (0, 3, 8):
      raise unittest.SkipTest(
          "`emit_python_callback` only supported in jaxlib >= 0.3.8")
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x, ordered=True)
    with capture_stdout() as output:
      f(2)
    self.assertEqual(output(), "x: 2\n")


if jaxlib.version < (0, 3, 8):
  # No lowering for `emit_python_callback` in older jaxlibs.
  del DebugPrintTest

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
