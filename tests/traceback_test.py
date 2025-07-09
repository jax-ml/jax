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

import contextlib
import traceback

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import _jax
from jax._src.lib import jaxlib_extension_version
import jax.numpy as jnp
import numpy as np

Traceback = _jax.Traceback
Frame = _jax.Frame


@contextlib.contextmanager
def tracebacks(enabled=True):
  """Context manager that enables or disables traceback collection."""
  saved = _jax.tracebacks_enabled()
  _jax.set_tracebacks_enabled(enabled)
  try:
    yield
  finally:
    _jax.set_tracebacks_enabled(saved)


class TracebackTest(absltest.TestCase):

  def testNoTracebacksIfDisabled(self):
    with tracebacks(enabled=False):
      self.assertEqual(None, Traceback.get_traceback())
      buffer = jnp.array(7, np.int32)
      self.assertEqual(None, buffer.traceback)

      e = jax.jit(lambda x: x + 1).lower(1).compile().runtime_executable()
      self.assertEqual(None, e.traceback)

  def assertIsTracebackContaining(self, tb, function):
    self.assertIsInstance(tb, Traceback)
    self.assertIn(function, str(tb))
    self.assertTrue(any(f.function_name == function for f in tb.frames))

  def testTracebacks(self):
    with tracebacks(enabled=True):
      if jaxlib_extension_version >= 362:
        fn = "TracebackTest.testTracebacks"
      else:
        fn = "testTracebacks"

      tb = Traceback.get_traceback()
      self.assertIsTracebackContaining(tb, fn)

      buffer = jnp.array(7, np.int32)
      self.assertIsTracebackContaining(buffer.traceback, fn)

      e = jax.jit(lambda x: x + 1).lower(1).compile().runtime_executable()
      self.assertIsTracebackContaining(e.traceback, fn)

  def testNestedFunction(self):

    def AFunction():

      def AnotherFunction():
        return Traceback.get_traceback()

      return AnotherFunction()

    with tracebacks(enabled=True):
      tb = AFunction()
      self.assertIsInstance(tb, Traceback)
      frames = tb.frames
      fn = (
          "TracebackTest.testNestedFunction.<locals>.AFunction"
          if jaxlib_extension_version >= 362
          else "AFunction"
      )
      i = next(i for (i, f) in enumerate(frames) if f.function_name == fn)
      if jaxlib_extension_version >= 362:
        self.assertEqual(
            frames[i - 1].function_name,
            "TracebackTest.testNestedFunction.<locals>.AFunction.<locals>.AnotherFunction",
        )
        self.assertEqual(
            frames[i + 1].function_name, "TracebackTest.testNestedFunction"
        )
      else:
        self.assertEqual(frames[i - 1].function_name, "AnotherFunction")
        self.assertEqual(frames[i + 1].function_name, "testNestedFunction")

  def testPythonTracebackHasCorrectLineNumbers(self):
    def B():
      return Traceback.get_traceback()

    def A():
      return B()

    tb = A().as_python_traceback()
    for frame, lineno in traceback.walk_tb(tb):
      if frame.f_code.co_name == "A":
        line = A.__code__.co_firstlineno
        self.assertBetween(lineno, line, line + 2)
      elif frame.f_code.co_name == "B":
        line = B.__code__.co_firstlineno
        self.assertBetween(lineno, line, line + 2)

  def testAccessingLocalsDoesNotCrash(self):
    # https://github.com/google/jax/issues/16027
    tb = Traceback.get_traceback()
    python_tb = tb.as_python_traceback()
    for frame, _ in traceback.walk_tb(python_tb):
      _ = frame.f_locals  # should not crash

  def testTracebackFromFrames(self):
    def FooFn(x):
      return x + 1

    def BarFn(y):
      y = y + 1
      y = y + 2
      return y * 2

    frame_foo = Frame(
        __file__,
        FooFn.__code__.co_name,
        FooFn.__code__.co_firstlineno,
        FooFn.__code__.co_firstlineno + 1,
    )
    frame_bar = Frame(
        __file__,
        BarFn.__code__.co_name,
        BarFn.__code__.co_firstlineno,
        BarFn.__code__.co_firstlineno + 2,
    )
    frames = [frame_foo, frame_bar]
    tb = Traceback.traceback_from_frames(frames)

    with self.subTest("WalkDoesNotError"):
      for frame, _ in traceback.walk_tb(tb):
        _ = frame.f_locals  # should not crash

    with self.subTest("TracebackCorrectness"):
      tb_string = traceback.format_tb(tb)
      # The traceback should have the format:
      # File <this file>, line N in BarFn
      #   y = y + 2
      # File <this file>, line N in FooFn
      #   return x + 1
      self.assertLen(tb_string, len(frames))
      bar_frame = tb_string[0].split("\n")
      self.assertEndsWith(bar_frame[0], "BarFn")
      self.assertEqual(bar_frame[1].strip(), "y = y + 2")
      foo_frame = tb_string[1].split("\n")
      self.assertEndsWith(foo_frame[0], "FooFn")
      self.assertEqual(foo_frame[1].strip(), "return x + 1")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
