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
- Test decorators that mark a test case or test class as thread-unsafe.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import threading
import time
import unittest

from absl.testing import absltest
from jax._src import config
from jax._src import test_warning_util

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




def thread_unsafe_test(condition: bool = True):
  """Decorator for tests that are not thread-safe.

  Args:
    condition: If True, mark the test as thread-unsafe. If False, the test
      may run in parallel with other tests. Defaults to True.
  """
  def decorator(func):
    setattr(func, "thread_unsafe", condition)
    return func
  return decorator


def thread_unsafe_test_class(condition: bool = True):
  """Decorator that marks a TestCase class as thread-unsafe.

  Args:
    condition: If True, mark the test class as thread-unsafe. If False, the
      test class runs normally. Defaults to True.
  """
  def f(klass):
    assert issubclass(klass, unittest.TestCase), type(klass)
    klass.thread_unsafe = condition
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
        self.test_result.time_getter = lambda: self.start_time  # pyrefly: ignore[missing-attribute]
        self.test_result.startTest(test)
        for callback in self.actions:
          callback()
        self.test_result.time_getter = lambda: stop_time  # pyrefly: ignore[missing-attribute]
        self.test_result.stopTest(test)
      finally:
        if time_getter is not None:
          self.test_result.time_getter = time_getter  # pyrefly: ignore[missing-attribute]

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


def _is_thread_unsafe(test: unittest.TestCase) -> bool:
  if not isinstance(test, unittest.TestCase):
    return False
  if getattr(test.__class__, "thread_unsafe", False):
    return True
  method = getattr(test, test._testMethodName)
  if getattr(method, "thread_unsafe", False):
    return True
  return False


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

    thread_safe_tests = []
    thread_unsafe_tests = []

    def partition_test(test):
      if isinstance(test, unittest.TestSuite):
        for subtest in test:
          partition_test(subtest)
      else:
        if _is_thread_unsafe(test):
          thread_unsafe_tests.append(test)
        else:
          thread_safe_tests.append(test)

    partition_test(self)

    with executor:
      for test in thread_safe_tests:
        test_result = ThreadSafeTestResult(lock, result)
        futures.append(executor.submit(test, test_result))

      for future in futures:
        future.result()

    for test in thread_unsafe_tests:
      test_result = ThreadSafeTestResult(lock, result)
      test(test_result)

    return result


class JaxTestLoader(absltest.TestLoader):
  suiteClass = JaxTestSuite  # pyrefly: ignore[bad-assignment]

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
