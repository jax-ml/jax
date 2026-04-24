# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import itertools
import logging
import os
import unittest

import hypothesis as hp
from hypothesis.internal import detection
from hypothesis.internal import reflection
from hypothesis.strategies._internal import core as hps_internal_core
from jax._src import config
from jax._src import test_util as jtu
from jax._src.test_loader import JaxTestLoader

HYPOTHESIS_PROFILE = config.string_flag(
    "hypothesis_profile",
    os.getenv("JAX_HYPOTHESIS_PROFILE", "deterministic"),
    help=(
        "Select the hypothesis profile to use for testing. Available values: "
        "deterministic, interactive"
    ),
)


_TEST_SHARD_INDEX = int(os.environ.get("TEST_SHARD_INDEX", "0"))
_TEST_TOTAL_SHARDS = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))


def hypothesis_inner_test_shard(inner_test, args, kwargs, total_shards):
  """Returns the expected shard index for a generated Hypothesis example."""
  text_repr = reflection.repr_call(inner_test, args, kwargs)
  test_hash = int(hashlib.md5(text_repr.encode()).hexdigest(), 16)
  return test_hash % total_shards


def _shard_aware_hypothesis_inner_test(inner_test):
  @functools.wraps(inner_test)
  def shard_aware_inner_test_fn(*args, **kwargs):
    if _TEST_TOTAL_SHARDS == 1:
      return inner_test(*args, **kwargs)

    mod = hypothesis_inner_test_shard(
        inner_test, args, kwargs, _TEST_TOTAL_SHARDS
    )
    if mod != _TEST_SHARD_INDEX:
      # We don't call `googletest.skip` here because skipping within hypothesis
      # will skip the *entire* test, not just the current example.
      return None
    else:
      return inner_test(*args, **kwargs)

  return shard_aware_inner_test_fn


def _shard_aware_test(test, shard_index):
  """Wraps a non-Hypothesis test to skip on the wrong shard.

  HypothesisShardedTestLoader bypasses absltest's normal round-robin sharding,
  so non-Hypothesis tests inside a HypothesisShardedTestCase would otherwise
  run on every shard. This wrapper reimplements round-robin at the method level.
  """

  @functools.wraps(test)
  def shard_aware_test_fn(*args, **kwargs):
    if _TEST_SHARD_INDEX != shard_index:
      return unittest.skip(f"Running on shard {shard_index}")(test)(
          *args, **kwargs
      )
    else:
      return test(*args, **kwargs)

  return shard_aware_test_fn


def _apply_sharding_to_tests(test_runner):

  shards_index_iter = itertools.cycle(range(_TEST_TOTAL_SHARDS))
  for name in dir(test_runner):
    if name.startswith("test"):
      test = getattr(test_runner, name)
      if detection.is_hypothesis_test(test):
        handle = test.hypothesis
        assert isinstance(handle, hp.core.HypothesisHandle)
        # `@given(..., data())` is not supported because:
        # - Sharding requires known values for all drawn parameters.
        # - The test body must be called given the sharding.
        # - Using `data()`, some or all parameters are not known until the test
        #   body is called.
        for val in handle._given_kwargs.values():
          if isinstance(val, hps_internal_core.DataStrategy):
            raise ValueError(
                "Sharded hypothesis runner does not support `data()` inside"
                " `@given`. All parameters must be drawn before the test body"
                " is called. Consider using `@composite` instead."
            )
        handle.inner_test = _shard_aware_hypothesis_inner_test(
            handle.inner_test
        )
      else:
        # If the tests are not hypothesis tests (or we are not sharding
        # hypothesis tests), we can just assign them to shards in a round-robin
        # fashion.
        shard_index = next(shards_index_iter)
        setattr(test_runner, name, _shard_aware_test(test, shard_index))


class HypothesisShardedTestCase(jtu.JaxTestCase):
  """Runs Hypothesis tests in a sharded manner.

  The way hypothesis works is that it will run the same test function with
  different arguments, and it will run the same test function with the same
  arguments multiple times.

  This makes it difficult for Bazel test sharding to work, as the test loader
  sees the test function as a single test — and will schedule it to a single
  shard. So a hypothesis test with 300 examples will run on a single shard, and
  will take a long time.

  This class works around this by running each hypothesis test on every single
  shard, and then filtering out the tests that don't belong to the current
  shard.

  Must be combined with `HypothesisShardedTestLoader`.
  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if _TEST_TOTAL_SHARDS > 1:
      _apply_sharding_to_tests(cls)


class HypothesisShardedTestLoader(JaxTestLoader):
  """A TestLoader that bypasses method-level sharding.

  Used with `jtu.HypothesisShardedTestCase` to implement inner-test sharding
  for slow hypothesis tests.
  """

  def getTestCaseNames(self, testCaseClass):
    self._current_test_class = testCaseClass
    return super().getTestCaseNames(testCaseClass)

  def shardTestCaseNames(self, iterator, ordered_names, shard_index):
    if issubclass(self._current_test_class, HypothesisShardedTestCase):
      return ordered_names
    return super().shardTestCaseNames(iterator, ordered_names, shard_index)


def hypothesis_is_thread_safe() -> bool:
  """Returns True if the installed hypothesis version is thread-safe.

  Hypothesis versions >= 6.136.9 are thread-safe.
  """
  return tuple(int(x) for x in hp.__version__.split(".")) >= (6, 136, 9)


def setup_hypothesis(max_examples=30) -> None:
  """Sets up the hypothesis profiles.

  Sets up the hypothesis testing profiles, and selects the one specified by
  the ``JAX_HYPOTHESIS_PROFILE`` environment variable (or the
  ``--jax_hypothesis_profile`` configuration.

  Args:
    max_examples: the maximum number of hypothesis examples to try, when using
      the default "deterministic" profile.
  """
  # In our tests we often use subclasses with slightly different class variables
  # to generate whole suites of parameterized tests, but this approach does not
  # work well with Hypothesis databases, which use some function of the method
  # identity to generate keys. But, if the method is defined in a superclass,
  # all subclasses share the same key. This key collision can lead to confusing
  # false positives in other health checks.
  #
  # Still, as far as I understand, for as long as we don't use the example
  # database, it should be perfectly safe to suppress this health check. This
  # seems simpler than rewriting our tests that trigger this behavior. See
  # the end of https://github.com/HypothesisWorks/hypothesis/issues/3446 for
  # more context.
  suppressed_checks = []
  if hasattr(hp.HealthCheck, "differing_executors"):
    suppressed_checks.append(hp.HealthCheck.differing_executors)
  if jtu.is_asan() or jtu.is_msan() or jtu.is_tsan():
    suppressed_checks.append(hp.HealthCheck.too_slow)

  hp.settings.register_profile(
      "deterministic",
      database=None,
      derandomize=True,
      deadline=None,
      max_examples=max_examples,
      print_blob=True,
      suppress_health_check=suppressed_checks,
  )
  hp.settings.register_profile(
      "interactive",
      parent=hp.settings.load_profile("deterministic"),
      max_examples=1,
      report_multiple_bugs=False,
      verbosity=hp.Verbosity.verbose,
      # Don't try and shrink
      phases=(
          hp.Phase.explicit,
          hp.Phase.reuse,
          hp.Phase.generate,
          hp.Phase.target,
          hp.Phase.explain,
      ),
  )
  profile = HYPOTHESIS_PROFILE.value
  logging.info("Using hypothesis profile: %s", profile)
  hp.settings.load_profile(profile)
