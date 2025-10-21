# Copyright 2018 The JAX Authors.
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

"""
Contains a custom unittest loader and test suite.

Implements:
- A test filter based on the JAX_TEST_TARGETS and JAX_EXCLUDE_TEST_TARGETS
  environment variables.
- A test suite that runs tests in parallel using threads if JAX_TEST_NUM_THREADS
  is >= 1.
- Test decorators that mark a test case or test class as thread-hostile.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import os
import re
import threading
import time
import unittest

from absl.testing import absltest
from jax._src import config
from jax._src import test_warning_util
from jax._src import util

logger = logging.getLogger(__name__)


_TEST_TARGETS = config.string_flag(
  'test_targets', os.getenv('JAX_TEST_TARGETS', ''),
  'Regular expression specifying which tests to run, called via re.search on '
  'the test name. If empty or unspecified, run all tests.'
)

_EXCLUDE_TEST_TARGETS = config.string_flag(
  'exclude_test_targets', os.getenv('JAX_EXCLUDE_TEST_TARGETS', ''),
  'Regular expression specifying which tests NOT to run, called via re.search '
  'on the test name. If empty or unspecified, run all tests.'
)

TEST_NUM_THREADS = config.int_flag(
    'jax_test_num_threads', int(os.getenv('JAX_TEST_NUM_THREADS', '0')),
    help='Number of threads to use for running tests. 0 means run everything '
    'in the main thread. Using > 1 thread is experimental.'
)

# We use a reader-writer lock to protect test execution. Tests that may run in
# parallel acquire a read lock; tests that are not thread-safe acquire a write
# lock.
_test_rwlock = util.Mutex()

def _run_one_test(test: unittest.TestCase, result: ThreadSafeTestResult):
  if getattr(test.__class__, "thread_hostile", False):
    _test_rwlock.writer_lock()
    try:
      test(result)  # type: ignore
    finally:
      _test_rwlock.writer_unlock()
  else:
    _test_rwlock.reader_lock()
    try:
      test(result)  # type: ignore
    finally:
      _test_rwlock.reader_unlock()


@contextmanager
def thread_unsafe_test():
  """Decorator for tests that are not thread-safe.

  Note: this decorator (naturally) only applies to what it wraps, not to, say,
  code in separate setUp() or tearDown() methods.
  """
  if TEST_NUM_THREADS.value <= 0:
    yield
    return

  _test_rwlock.assert_reader_held()
  _test_rwlock.reader_unlock()
  _test_rwlock.writer_lock()
  try:
    yield
  finally:
    _test_rwlock.writer_unlock()
    _test_rwlock.reader_lock()


def thread_unsafe_test_class():
  """Decorator that marks a TestCase class as thread-hostile."""
  def f(klass):
    assert issubclass(klass, unittest.TestCase), type(klass)
    klass.thread_hostile = True
    return klass
  return f


class ThreadSafeTestResult:
  """
  Wraps a TestResult to make it thread safe.

  We do this by accumulating API calls and applying them in a batch under a
  lock at the conclusion of each test case.

  We duck type instead of inheriting from TestResult because we aren't actually
  a perfect implementation of TestResult, and would rather get a loud error
  for things we haven't implemented.
  """
  def __init__(self, lock: threading.Lock, result: unittest.TestResult):
    self.lock = lock
    self.test_result = result
    self.actions: list[Callable[[], None]] = []

  def startTest(self, test: unittest.TestCase):
    logger.info("Test start: %s", test.id())
    self.start_time = time.time()

  def stopTest(self, test: unittest.TestCase):
    logger.info("Test stop: %s", test.id())
    stop_time = time.time()
    with self.lock:
      # If test_result is an ABSL _TextAndXMLTestResult we override how it gets
      # the time. This affects the timing that shows up in the XML output
      # consumed by CI.
      time_getter = getattr(self.test_result, "time_getter", None)
      try:
        self.test_result.time_getter = lambda: self.start_time
        self.test_result.startTest(test)
        for callback in self.actions:
          callback()
        self.test_result.time_getter = lambda: stop_time
        self.test_result.stopTest(test)
      finally:
        if time_getter is not None:
          self.test_result.time_getter = time_getter

  def addSuccess(self, test: unittest.TestCase):
    self.actions.append(lambda: self.test_result.addSuccess(test))

  def addSkip(self, test: unittest.TestCase, reason: str):
    self.actions.append(lambda: self.test_result.addSkip(test, reason))

  def addError(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addError(test, err))

  def addFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addFailure(test, err))

  def addExpectedFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addExpectedFailure(test, err))

  def addDuration(self, test: unittest.TestCase, elapsed):
    self.actions.append(lambda: self.test_result.addDuration(test, elapsed))


class JaxTestSuite(unittest.TestSuite):
  """Runs tests in parallel using threads if TEST_NUM_THREADS is > 1.

  Caution: this test suite does not run setUpClass or setUpModule methods if
  thread parallelism is enabled.
  """

  def __init__(self, suite: unittest.TestSuite):
    super().__init__(list(suite))

  def run(self, result: unittest.TestResult, debug: bool = False) -> unittest.TestResult:
    if TEST_NUM_THREADS.value <= 0:
      return super().run(result)

    test_warning_util.install_threadsafe_warning_handlers()

    executor = ThreadPoolExecutor(TEST_NUM_THREADS.value)
    lock = threading.Lock()
    futures = []

    def run_test(test):
      """Recursively runs tests in a test suite or test case."""
      if isinstance(test, unittest.TestSuite):
        for subtest in test:
          run_test(subtest)
      else:
        test_result = ThreadSafeTestResult(lock, result)
        futures.append(executor.submit(_run_one_test, test, test_result))

    with executor:
      run_test(self)
      for future in futures:
        future.result()

    return result


class JaxTestLoader(absltest.TestLoader):
  suiteClass = JaxTestSuite

  def getTestCaseNames(self, testCaseClass):
    names = super().getTestCaseNames(testCaseClass)
    if _TEST_TARGETS.value:
      pattern = re.compile(_TEST_TARGETS.value)
      names = [name for name in names
               if pattern.search(f"{testCaseClass.__name__}.{name}")]
    if _EXCLUDE_TEST_TARGETS.value:
      pattern = re.compile(_EXCLUDE_TEST_TARGETS.value)
      names = [name for name in names
               if not pattern.search(f"{testCaseClass.__name__}.{name}")]
    return names
