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
from __future__ import annotations

import contextlib
import os
from typing import Any, Callable
import unittest
from unittest import mock

from jax._src import test_util as jtu
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import repro
from jax._src.repro import tracker
from jax._src.repro import emitter
from jax._src.pallas.mosaic import tpu_info


def maybe_skip_known_failure(msg: str = ""):
  if os.getenv("SKIP_KNOWN_FAILURES", ""):
    raise unittest.SkipTest(f"TODO: SKIP_KNOWN_FAILURES is set: {msg}")


@contextlib.contextmanager
def mock_tpu_context(
    chip_version: tpu_info.ChipVersion = tpu_info.ChipVersion.TPU_V6E,
    num_cores: int = 1,
):
  """Returns a context manager or test decorator that mocks TPU info
  for non-TPU devices.

  This can be used with jax.export for tests on CPU.
  """
  fake_info = tpu_info._get_tpu_info_impl(chip_version, num_cores)
  if jtu.is_device_tpu_at_least(fake_info.generation):
    yield
    return
  with mock.patch(
      "jax._src.pallas.mosaic.tpu_info.get_tpu_info", return_value=fake_info
  ), mock.patch(
      "jax._src.pallas.mosaic.tpu_info.is_tpu_device", return_value=True
  ):
    yield


def flatten_custom_pytree(t) -> Any:
  # The repro results contain custom pytree nodes flattened to a tuple. We
  # do the same to the direct results so that we can compare them.
  if isinstance(t, tuple) and not hasattr(t, "_fields"):  # Not a NamedTuple
    return tuple(flatten_custom_pytree(e) for e in t)
  if isinstance(t, list):
    return [flatten_custom_pytree(e) for e in t]
  if isinstance(t, dict):
    return {k: flatten_custom_pytree(e) for k, e in t.items()}
  if isinstance(t, tree_util.Partial):
    return t
  if t is None:
    return t
  # We want to eliminate the custom pytrees, but not the ones that JAX uses
  # internally.
  leaves = tree_util.tree_leaves(t)
  return leaves[0] if len(leaves) == 1 else leaves

class ReproTestBase(jtu.JaxTestCase):
  def setUp(self):
    if not traceback_util.repro_is_enabled():
      self.skipTest("JAX_REPRO_DIR not set")
    repro.reset_last_saved_repro()
    tracker._thread_local_state.reset_counters()
    self.repro_name_prefix = f"{self._testMethodName.removeprefix('test_')}_repro"
    self.enter_context(tracker.flags_override(error_mode="raise_tracking"))
    super().setUp()

  def assert_empty_call_stack(self):
    self.assertLen(tracker._thread_local_state.call_stack, 0)

  def collect_and_check(self, func: Callable, *args,
                        collect_static_argnums=(),
                        collect_static_argnames=(),
                        expect_exception: tuple[type[Exception], str] | None = None,
                        skip_repro_read: bool = False,
                        skip_repro_eval: bool = False,
                        atol=None, rtol=None,
                        **kwargs) -> str | None:
    """Runs `func` and verifies that result is the same when calling directly
       and calling the repro.
    """
    direct_result = None
    if expect_exception:
      context = self.assertRaisesRegex(* expect_exception)
    else:
      context = contextlib.nullcontext()

    with context:
      col = repro.collector(func, static_argnums=collect_static_argnums,
                            static_argnames=collect_static_argnames)
      try:
        direct_result = col(*args, **kwargs)  # Fails here if test is broken even without repro
      finally:
        source = col.to_source()
        emitter.save(source, repro_name_prefix=self.repro_name_prefix)
        if col.deferred_error is not None:
          raise col.deferred_error

    if skip_repro_read:
      return None
    repro_path, repro_source = repro.last_saved_repro()  # type: ignore
    if skip_repro_eval:
      return repro_source

    with tracker.enable(False):
      main_repro = repro.load(repro_source, repro_path)
      with context:
        repro_result = main_repro.run()

    if not expect_exception:
      self.assertAllClose(repro_result,
                          flatten_custom_pytree(direct_result),
                          atol=atol, rtol=rtol)
    return repro_source
