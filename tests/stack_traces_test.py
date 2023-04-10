# Copyright 2020 The JAX Authors.
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

import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
import jax._src.test_util as jtu
from absl.testing import absltest
from jax.config import config
from textwrap import dedent

config.parse_flags_with_absl()


class StackTracesTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    jax_dir = '/'.join(os.path.dirname(__file__).split('/')[:-1])
    self.binary_file = os.path.join(jax_dir, 'tests/stack_traces_test_util')
    self.filename = os.path.join(jax_dir, 'tests/stack_traces_test_util.py')

  @unittest.skipIf(not hasattr(signal, 'SIGSEGV'), 'Missing signal.SIGSEGV')
  def testSigsegv(self):
    traces_dir = tempfile.mkdtemp()
    error = 'Fatal Python error: Segmentation fault'
    self.check_fatal_error(traces_dir, 26, error, 'SIGSEGV')

  @unittest.skipIf(not hasattr(signal, 'SIGABRT'), 'Missing signal.SIGABRT')
  def testSigabrt(self):
    traces_dir = tempfile.mkdtemp()
    error = 'Fatal Python error: Aborted'
    self.check_fatal_error(traces_dir, 29, error, 'SIGABRT')

  @unittest.skipIf(not hasattr(signal, 'SIGFPE'), 'Missing signal.SIGFPE')
  def testSigfpe(self):
    traces_dir = tempfile.mkdtemp()
    error = 'Fatal Python error: Floating point exception'
    self.check_fatal_error(traces_dir, 32, error, 'SIGFPE')

  @unittest.skipIf(not hasattr(signal, 'SIGILL'), 'Missing signal.SIGILL')
  def testSigill(self):
    traces_dir = tempfile.mkdtemp()
    error = 'Fatal Python error: Illegal instruction'
    self.check_fatal_error(traces_dir, 35, error, 'SIGILL')

  @unittest.skipIf(not hasattr(signal, 'SIGBUS'), 'Missing signal.SIGBUS')
  def testSigbus(self):
    traces_dir = tempfile.mkdtemp()
    error = 'Fatal Python error: Bus error'
    self.check_fatal_error(traces_dir, 38, error, 'SIGBUS')

  @unittest.skipIf(not hasattr(signal, 'SIGUSR1'), 'Missing signal.SIGUSR1')
  def testSigusr(self):
    traces_dir = tempfile.mkdtemp()
    self.check_fatal_error(traces_dir, 41, '', 'SIGUSR1')

  def check_fatal_error(self, traces_dir, line_number, error, signal):
    """Check the traceback from the child process.

    Raise an error if the traceback doesn't match the expected format.
    """
    header = r'Current thread XXX \(most recent call first\)'
    if error:
      regex = """
          {error}

          {header}:
            File "{filename}", line {line_number} in <module>
          """
    else:
      regex = """
          {header}:
            File "{filename}", line {line_number} in <module>
          """
    regex = (
        dedent(regex)
        .format(
            error=error,
            header=header,
            filename=self.filename,
            line_number=line_number,
        )
        .strip()
    )
    output = self.get_output(traces_dir, signal)
    self.assertRegex(output, regex)

  def run_python_code(self, signal, traces_dir):
    args = ['--signal=' + signal, '--traces_dir=' + traces_dir]
    if sys.executable is not None:
      code = [sys.executable, self.filename]
    else:
      code = [self.binary_file]
    return subprocess.Popen(
        code + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

  def get_output(self, traces_dir, signal):
    """Run the specified code and read the output."""
    process = self.run_python_code(signal, traces_dir)
    stdout, _ = process.communicate()
    output = re.sub(rb'\[\d+ refs\]\r?\n?', b'', stdout).strip()
    output = output.decode('ascii', 'backslashreplace')
    traces_file = os.listdir(traces_dir + '/debugging')
    if traces_file:
      self.assertEqual(output, '')
      stack_traces_file = traces_dir + '/debugging/' + traces_file[0]
      with open(stack_traces_file, 'rb') as fp:
        output = fp.read()
      output = output.decode('ascii', 'backslashreplace')
    output = re.sub('Current thread 0x[0-9a-f]+', 'Current thread XXX', output)
    return output


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
