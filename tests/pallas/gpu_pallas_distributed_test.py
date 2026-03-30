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

"""Tests for distributed pallas GPU operations."""

import dataclasses
import functools
import os
import tempfile
import types
from typing import ClassVar
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax._src.config import config
from jax._src.lib import cuda_versions
from jax.experimental import multihost_utils
from jax.experimental import pallas as _pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as _plgpu
from jax.experimental.pallas.ops.gpu.all_gather_mgpu import all_gather
from jax.experimental.pallas.ops.gpu.reduce_scatter_mgpu import reduce_scatter
import jax.numpy as jnp
import numpy as np


P = jax.sharding.PartitionSpec
partial = functools.partial


# We don't want the user to call pl.pallas_call or plgpu.kernel directly, so we
# monkey patch the functions in `pl` and `plgpu`.
def do_not_call_me_directly(*args, **kwargs):
  raise RuntimeError(
      "Use self.{kernel,pallas_call} instead of {plgpu.kernel,pl.pallas_call}."
  )

# Clone the modules locally because the functions are called from other
# other test files in OSS and the tests are not isolated.
pl = types.ModuleType("_pl_local")
pl.__dict__.update(_pl.__dict__)

plgpu = types.ModuleType("_plgpu_local")
plgpu.__dict__.update(_plgpu.__dict__)

_pallas_call = _pl.pallas_call
_kernel = _plgpu.kernel
del _pl, _plgpu

plgpu.kernel = do_not_call_me_directly
pl.pallas_call = do_not_call_me_directly


def is_nvshmem_used():
  return (
      "XLA_FLAGS" in os.environ
        and "--xla_gpu_experimental_enable_nvshmem" in os.environ["XLA_FLAGS"])


def get_reduction_impl(reduction):
  match reduction:
    case "add":
      return jnp.add
    case "min":
      return jnp.minimum
    case "max":
      return jnp.maximum
    case "and":
      return jnp.bitwise_and
    case "or":
      return jnp.bitwise_or
    case "xor":
      return jnp.bitwise_xor
    case _:
      raise ValueError(reduction)


_TestCaseBase = (jt_multiprocess.MultiProcessTest
                 if is_nvshmem_used() is None
                 else parameterized.TestCase)


class MonkeyPatchTest:
  def test_calling_pallas_call_directly_raises(self):
    with self.assertRaises(RuntimeError):
      pl.pallas_call()

  def test_calling_kernel_directly_raises(self):
    with self.assertRaises(RuntimeError):
      plgpu.kernel()


class PallasTestMetaclass(type(_TestCaseBase)):

  def __new__(mcs, *args, lowering_semantics=plgpu.LoweringSemantics.Lane):
    cls = super().__new__(mcs, *args)
    cls.LOWERING_SEMANTICS = lowering_semantics
    return cls


class TestCase(_TestCaseBase, metaclass=PallasTestMetaclass):
  LOWERING_SEMANTICS: ClassVar[plgpu.LoweringSemantics]
  # We track whether we called monkey patched APIs for every test.
  # In tearDown, we verify that the test actually called the monkey-patched APIs.
  # This is not a perfect way of ensuring we do not call plgpu.kernel and
  # plgpu.pallas_call, but is a helpful safeguard.
  monkey_patched_api_was_used: bool

  def setUp(self):
    if jtu.test_device_matches(["rocm"]):
      self.skipTest("Mosaic GPU is not supported on ROCm.")

    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    if not mgpu.supports_cross_device_collectives():
      self.skipTest(
          "Skip test since cross-device collectives are not supported"
          " (either NVSHMEM is not available in multi-process mode, or mixed"
          " mode is used).")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "") == "platform":
      self.skipTest("NVSHMEM doesn't work with the platform allocator.")

    # TODO(b/415721295): remove this check once minimum jaxlib version is 0.10.0.
    # "mgpu.dialect has a backward incompatible change."
    if not hasattr(mgpu.dialect, "WarpMapOp"):
      self.skip_if_wg_semantics()

    self.monkey_patched_api_was_used = False
    super().setUp()

  def tearDown(self):
    self.assertTrue(self.monkey_patched_api_was_used)

  def is_wg_semantics(self):
    return self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup

  def skip_if_wg_semantics(self):
    if self.is_wg_semantics():
      self.skipTest("Not supported under WG semantics")

  def pallas_call(self, *args, **kwargs):
    compiler_params = dataclasses.replace(
        kwargs.pop("compiler_params", plgpu.CompilerParams()),
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    result = _pallas_call(*args, compiler_params=compiler_params, **kwargs)
    self.monkey_patched_api_was_used = True
    return result

  def kernel(self, *args, **kwargs):
    compiler_params = dataclasses.replace(
        kwargs.pop("compiler_params", plgpu.CompilerParams()),
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    result = _kernel(*args, compiler_params=compiler_params, **kwargs)
    self.monkey_patched_api_was_used = True
    return result

  def skipTest(self, msg):
    # Setting `monkey_patched_api_was_used` to true for skipped tests to prevent
    # the assertion failure on teardown.
    self.monkey_patched_api_was_used = True
    super().skipTest(msg)


class PallasCallRemoteDMATest(TestCase):
  def setUp(self):
    if jax.device_count() < 2:
      self.skipTest("Needs at least two devices")
    super().setUp()

  def test_remote_dma_basic(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = 1 - lax.axis_index('x')
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return self.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
          compiler_params=plgpu.CompilerParams(),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ["x"])
    y = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),
            check_vma=False,
        )
    )(x)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  # Test verifies an execution of HLO with several slightly different mosaic
  # custom calls. The difference is needed to validate correct initialization
  # of the collective metadata before each kernel execution.
  def test_remote_dma_several_mosaic_ops(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(x_ref, y_ref, done_sem):
      other_dev_id = 1 - lax.axis_index("x")
      pl.semaphore_signal(done_sem, device_id=other_dev_id)
      pl.semaphore_wait(done_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(done_sem, device_id=other_dev_id)
      pl.semaphore_wait(done_sem)

    def different_kernel(x_ref, y_ref, wait_sem, ready_sem):
      other_dev_id = 1 - lax.axis_index("x")
      pl.semaphore_signal(wait_sem, device_id=other_dev_id)
      pl.semaphore_wait(wait_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)

    def body(x):
      result = x
      for _ in range(25):
        result = self.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            scratch_shapes=[
                plgpu.SemaphoreType.REGULAR,
            ],
        )(result)

        result = self.pallas_call(
            different_kernel,
            in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            scratch_shapes=[
                plgpu.SemaphoreType.REGULAR,
                plgpu.SemaphoreType.REGULAR,
            ],
        )(result)

      return result

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ["x"])
    y = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),
            check_vma=False,
        )
    )(x)

    expected = x[:8] if jax.process_index() == 0 else x[8:]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_remote_dma_with_retries(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = 1 - lax.axis_index('x')
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return self.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
          compiler_params=plgpu.CompilerParams(),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    jit_body = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
            check_vma=False,
        )
    )
    y = x
    for _ in range(51):
      y = jit_body(y)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_remote_dma_with_profiler(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = 1 - lax.axis_index('x')
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    # Ignore the warning about the profile already existing since in a
    # single-process mode both device results will try to write to the same
    # profile file.
    with jtu.ignore_warning(category=UserWarning,
                            message=".*profile already exists.*"):
      with tempfile.TemporaryDirectory() as tmpdir:
        x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
        def body(x):
          return self.pallas_call(
              kernel,
              in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
              out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
              out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
              scratch_shapes=[
                  plgpu.SemaphoreType.REGULAR,
                  plgpu.SemaphoreType.REGULAR,
              ],
              compiler_params=plgpu.CompilerParams(
                  profile_space=1, profile_dir=tmpdir
              ),
          )(x)

        devices = jax.devices()[:2]
        mesh = jax.sharding.Mesh(devices, ['x'])
        y = jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False,
            )
        )(x)

      expected = x[8:] if jax.process_index() == 0 else x[:8]
      np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_remote_dma_in_loop(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      device_id = lax.axis_index('x')
      other_dev_id = 1 - device_id
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      def body(i, _):
        y_ref.at[0, i].set(x_ref.at[0, i].get())
        pl.semaphore_signal(ready_sem, device_id=other_dev_id)
        pl.semaphore_wait(ready_sem)
        neighbor_ptr.at[0, i].set(x_ref.at[0, i].get())
        pl.semaphore_signal(recv_sem, device_id=other_dev_id)
        pl.semaphore_wait(recv_sem)

      lax.fori_loop(0, 128, body, init_val=None, unroll=False)

    x = jnp.arange(2 * 128.0, dtype=jnp.float32).reshape((2, 128))
    def body(x):
      return self.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
          compiler_params=plgpu.CompilerParams(),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False,
        )
    )(x)

    expected = x[1:] if jax.process_index() == 0 else x[:1]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_remote_dma_dynamic_index(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = jnp.sum(x_ref[...] == 1, dtype=jnp.int32)
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.zeros((2, 128), dtype=jnp.int32)
    x = x.at[0, 0].set(1)
    def body(x):
      return self.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((1, 128), jnp.int32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
          compiler_params=plgpu.CompilerParams(),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False,
        )
    )(x)

    expected = x[1:] if jax.process_index() == 0 else x[:1]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  @parameterized.parameters(('x',), ('y',))
  def test_remote_dma_2d_mesh(self, axis):
    if (jax.process_count() < 4 and
        (jax.process_count() != 1 or jax.local_device_count() < 4)):
      self.skipTest('Test requires at least 4 devices either accessible by a'
                    'single process or 4 processes with 1 device each.')
    if jax.process_index() > 4:
      return  # Only 4 processes needed.
    def kernel(x_ref, y_ref, recv_sem):
      other_dev_id = {axis: 1 - lax.axis_index(axis)}
      other_y_ref = plgpu.remote_ref(y_ref, other_dev_id)
      other_y_ref[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return self.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[plgpu.SemaphoreType.REGULAR],
          compiler_params=plgpu.CompilerParams(),
      )(x)

    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(np.asarray(devices).reshape(2, 2), ['x', 'y'])
    y = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=P(axis), out_specs=P(axis), check_vma=False,
        )
    )(x)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_wait_twice(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 2, device_id=other_dev_id)
      pl.semaphore_wait(sem)
      pl.semaphore_wait(sem)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        compiler_params=plgpu.CompilerParams(),
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P(None), check_vma=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_wait_nodec(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 2, device_id=other_dev_id)
      pl.semaphore_wait(sem, decrement=False)
      pl.semaphore_wait(sem, 2, decrement=False)
      pl.semaphore_wait(sem, 2)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        compiler_params=plgpu.CompilerParams(),
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P(None), check_vma=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_signal_parallel(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem, sem2):
      other_dev_id = 1 - lax.axis_index('x')
      plgpu.semaphore_signal_parallel(
          plgpu.SemaphoreSignal(sem, device_id=other_dev_id),
          plgpu.SemaphoreSignal(sem2, device_id=other_dev_id),
      )
      pl.semaphore_wait(sem)
      pl.semaphore_wait(sem2)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR] * 2,
        compiler_params=plgpu.CompilerParams(),
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P(None), check_vma=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_semaphore_signal_collective_axes(self):
    # TODO(b/476264413): Support multimem in multi-thread mode.
    if jax.local_device_count() > 1:
      self.monkey_patched_api_was_used = True
      return  # Multimem not supported in multi-thread mode yet.

    if jax.process_index() > 2:
      self.monkey_patched_api_was_used = True
      return  # Only 2 processes needed.

    def kernel(y_ref, sem_out_ref, sem):
      plgpu.semaphore_signal_multicast(sem.at[0], collective_axes='x')
      # Wait for the multicast signal (each device gets signaled by all devices)
      pl.semaphore_wait(sem.at[0], 2)  # Wait for signals from both devices
      y_ref[...] = jnp.ones_like(y_ref)
      sem_out_ref[0] = pl.semaphore_read(sem.at[0])
      sem_out_ref[1] = pl.semaphore_read(sem.at[1])

    kernel_call = self.pallas_call(
        kernel,
        out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)] * 2,
        out_shape=(
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR((2,))],
        compiler_params=plgpu.CompilerParams(),
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y, sem_out = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P(None), check_vma=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

    try:
      np.testing.assert_allclose(sem_out, jnp.zeros_like(sem_out))
    except Exception:
      # On some CUDA versions there is a compiler bug where the predicate
      # on the multimem reduction is not respected.
      if cuda_versions.cuda_runtime_get_version() not in [12080, 12090, 13000]:
        raise

  def test_permuted_mesh(self):
    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        compiler_params=plgpu.CompilerParams(),
    )
    mesh = jax.sharding.Mesh(jax.devices()[::-1], ['x'])  # Reverse the devices.
    f = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P(None), check_vma=False,
        )
    )
    msg = (
        'Mosaic GPU only supports meshes with device ordering that follows'
        ' row-major device ids.'
    )
    with self.assertRaisesRegex(NotImplementedError, msg):
      f()

  @parameterized.parameters(False, True)
  def test_copy_tma(self, use_dict):
    # TODO(bchetioui): support for remote refs.
    self.skip_if_wg_semantics()
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, smem_ref, sem):
      dev_id = lax.axis_index("y")
      other_dev_id = 1 - dev_id
      if use_dict:
        ids = lambda x, y: dict(x=x, y=y)
      else:
        ids = lambda x, y: (x, y)

      # Device ID must be an int32.
      zero = jnp.int32(0)

      @pl.when(dev_id == zero)
      def _store():
        output = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (128, 128), 1), plgpu.Layout.WGMMA)
        smem_ref[...] = output
        plgpu.copy_smem_to_gmem(smem_ref, plgpu.remote_ref(y_ref, ids(zero, dev_id)))
        plgpu.copy_smem_to_gmem(smem_ref, plgpu.remote_ref(y_ref, ids(zero, other_dev_id)))
        plgpu.wait_smem_to_gmem(0)
      pl.semaphore_signal(sem, 1, device_id=ids(zero, other_dev_id))
      pl.semaphore_wait(sem)

    transforms = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128))
    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        scratch_shapes=[
            plgpu.SMEM((128, 128), jnp.int32, transforms=transforms),
            plgpu.SemaphoreType.REGULAR,
        ],
        compiler_params=plgpu.CompilerParams(),
    )
    mesh = jtu.create_mesh((1, 2), ("x", "y"))
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P("y"), check_vma=False,
        )
    )()
    if jax.local_device_count() == 1:
      y = multihost_utils.process_allgather(y, tiled=True)
    ref = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    np.testing.assert_array_equal(y, np.concat([ref, ref], axis=0))


class PallasCallMultimemTest(TestCase):
  def setUp(self):
    if jax.local_device_count() > 1:
      self.skipTest("Multimem not supported in multi-thread mode yet.")
    if jax.device_count() < 2:
      self.skipTest("Needs at least two devices")
    # TODO(belitskiy): Remove the hasattr guard once JAX 0.9.2 is released.
    if not hasattr(cuda_versions, "cuda_supports_multicast"):
      self.skipTest("Multicast not yet supported")
    if any(
      not cuda_versions.cuda_supports_multicast(d.local_hardware_id)
      for d in jax.local_devices()
    ):
      self.skipTest("Not all local devices support multicast")
    super().setUp()

  def test_multimem_store_regs(self):
    # TODO(bchetioui): support for multimem store.
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      @pl.when(lax.axis_index('x') == 0)
      def _store():
        output = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (128, 128), 1), plgpu.Layout.WGMMA)
        plgpu.multimem_store(output, y_ref, 'x')
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        compiler_params=plgpu.CompilerParams(),
    )
    mesh = jax.sharding.Mesh(jax.devices(), ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P("x"), check_vma=False,
        )
    )()
    y = multihost_utils.process_allgather(y, tiled=True)
    ref = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    np.testing.assert_array_equal(y, np.concat([ref, ref], axis=0))

  def test_multimem_store_scalar(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      @pl.when(lax.axis_index('x') == 0)
      def _store():
        plgpu.multimem_store(1, y_ref.at[0], 'x')
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    kernel_call = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        compiler_params=plgpu.CompilerParams(),
    )
    mesh = jax.sharding.Mesh(jax.devices(), ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P('x'), check_vma=False,
        )
    )()
    y = multihost_utils.process_allgather(y, tiled=True)
    ref = jnp.array((1,), jnp.int32)
    np.testing.assert_array_equal(y, np.concat([ref, ref], axis=0), strict=True)

  def test_multimem_store_tma(self):
    # TODO(bchetioui): support for multimem store.
    self.skip_if_wg_semantics()
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, smem_ref, sem):
      @pl.when(lax.axis_index('x') == 0)
      def _store():
        output = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (128, 128), 1), plgpu.Layout.WGMMA)
        smem_ref[...] = output
        plgpu.copy_smem_to_gmem(smem_ref, plgpu.multicast_ref(y_ref, 'x'))
        plgpu.wait_smem_to_gmem(0)
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    transforms = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128))
    kernel_call = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        scratch_shapes=[
            plgpu.SMEM((128, 128), jnp.int32, transforms=transforms),
            plgpu.SemaphoreType.REGULAR,
        ],
        compiler_params=plgpu.CompilerParams(),
    )
    mesh = jax.sharding.Mesh(jax.devices(), ['x'])
    y = jax.jit(
        jax.shard_map(
            kernel_call, mesh=mesh, in_specs=(), out_specs=P("x"), check_vma=False,
        )
    )()
    y = multihost_utils.process_allgather(y, tiled=True)
    ref = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    np.testing.assert_array_equal(y, np.concat([ref, ref], axis=0))

  @parameterized.parameters(
      (jnp.int32, 1, "add"),
      (jnp.int32, 1, "min"),
      (jnp.int32, 1, "max"),
      (jnp.int32, 1, "and"),
      (jnp.int32, 1, "or"),
      (jnp.int32, 1, "xor"),
      (jnp.float32, 1, "add"),
      (jnp.float32, 2, "add", True),
      (jnp.float32, 4, "add"),
      (jnp.float16, 2, "add"),
      (jnp.float16, 2, "min"),
      (jnp.float16, 4, "max"),
      (jnp.float16, 8, "add", True),
      (jnp.bfloat16, 2, "max"),
      (jnp.bfloat16, 8, "add"),
      (jnp.float8_e5m2, 4, "add"),
      (jnp.float8_e5m2, 8, "min"),
      (jnp.float8_e5m2, 16, "max", True),
      (jnp.float8_e4m3fn, 4, "min", True),
      (jnp.float8_e4m3fn, 8, "max"),
      (jnp.float8_e4m3fn, 16, "add"),
  )
  def test_multimem_load_reduce(self, dtype, vector_length, reduction, tiled_layout=False):
    # TODO(bchetioui): support for multimem load reduce.
    self.skip_if_wg_semantics()
    if dtype in (
        jnp.float8_e5m2,
        jnp.float8_e4m3fn,
    ) and not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on GPU with capability >= sm100")
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    devices = jax.devices()[:2]

    def kernel(x_ref, y_ref, sem_ref):
      if tiled_layout:
        layout = plgpu.Layout.TILED(
            plgpu.Tiling(
                (
                    (64, 2 * vector_length),
                    (16, 2 * vector_length),
                    (vector_length,),
                )
            ),
            warp_dims=(-5,),
            lane_dims=(-3, -2),
            vector_dim=-1,
        )
      else:
        layout = plgpu.Layout.WG_STRIDED((64, 32), vec_size=vector_length)
      y_ref[...] = plgpu.layout_cast(
          plgpu.multimem_load_reduce(
              x_ref.at[16:-16], collective_axes="x", reduction_op=reduction,
          ),
          layout
      )
      my_device = lax.axis_index("x")
      other_device = 1 - my_device
      pl.semaphore_signal(sem_ref, 1, device_id=other_device)
      pl.semaphore_wait(sem_ref)

    # The rounding we see in low precision types seems to be different from
    # what JAX/XLA use.
    match jnp.dtype(dtype).itemsize:
      case 4:
        bound = 800000
      case 2:
        bound = 128
      case 1:
        bound = 4
      case _:
        raise ValueError(f"Unsupported dtype: {dtype}")
    x_local = jax.random.randint(
        jax.random.key(1234), (128 + 64, 32), dtype=jnp.int32, minval=-bound, maxval=bound,
    ).astype(dtype)
    mesh = jax.sharding.Mesh(devices, ("x",))
    y_shape = jax.ShapeDtypeStruct((64, 32), dtype)
    y = jax.jit(
        jax.shard_map(
            self.pallas_call(
                kernel,
                in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
                out_specs=pl.BlockSpec(memory_space=plgpu.SMEM),
                out_shape=y_shape,
                scratch_shapes=[plgpu.SemaphoreType.REGULAR],
                compiler_params=plgpu.CompilerParams(),
            ),
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),  # Not really, but lets us test.
            check_vma=False,
        )
    )(x_local)
    y = multihost_utils.process_allgather(y, tiled=True)
    np_reduction = get_reduction_impl(reduction)
    np.testing.assert_array_equal(
        y.astype(jnp.float32),
        np.tile(np_reduction(x_local[16:64+16], x_local[64+48:128+48]), (2, 1)),
    )


@jtu.thread_unsafe_test_class()
class PallasCallMultimemThreadUnsafeTest(TestCase):
  """
  This class is thread-unsafe because the tests monkey patch loaded modules.
  """

  def setUp(self):
    if jax.local_device_count() > 1:
      self.skipTest("Multimem not supported in multi-thread mode yet.")
    if jax.device_count() < 2:
      self.skipTest("Needs at least two devices")
    # TODO(belitskiy): Remove the hasattr guard once JAX 0.9.2 is released.
    if not hasattr(cuda_versions, "cuda_supports_multicast"):
      self.skipTest("Multicast not yet supported")
    if any(
      not cuda_versions.cuda_supports_multicast(d.local_hardware_id)
      for d in jax.local_devices()
    ):
      self.skipTest("Not all local devices support multicast")
    super().setUp()

  def _test_reduce_scatter(
      self,
      shape,
      dtype,
      reduction,
      scatter_dimension=0,
      tile_size=None,
      vec_size=None,
      num_blocks=None,
  ):
    self.skip_if_wg_semantics()  # Support multimem_load_reduce under WG.
    if jax.process_index() > 2:
      return

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ["x"])
    if jnp.issubdtype(dtype, jnp.floating):
      x = jax.random.uniform(jax.random.key(42), shape, dtype=dtype, minval=-1.0, maxval=1.0)
    else:
      x = jax.random.randint(jax.random.key(42), shape, dtype=dtype, minval=-1000, maxval=1000)

    def body(x):
      return reduce_scatter(
          x,
          axis_name="x",
          scatter_dimension=scatter_dimension,
          reduction=reduction,
          vec_size=vec_size,
          tile_size=tile_size,
          num_blocks=num_blocks,
      )

    spec = P(*([None] * scatter_dimension), "x")
    with mock.patch.object(
        jax.experimental.pallas.ops.gpu.reduce_scatter_mgpu.plgpu,
        "kernel",
        self.kernel,
    ):
      y = jax.jit(
          jax.shard_map(
              body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
          )
      )(x)

    y = multihost_utils.process_allgather(y, tiled=True)
    np_reduction = get_reduction_impl(reduction)

    split_idx = x.shape[scatter_dimension] // 2
    slices_first = [slice(None)] * len(shape)
    slices_first[scatter_dimension] = slice(None, split_idx)
    slices_second = [slice(None)] * len(shape)
    slices_second[scatter_dimension] = slice(split_idx, None)
    expected = np_reduction(x[tuple(slices_first)], x[tuple(slices_second)])
    tol = 1e-5 if reduction == "add" else 0
    np.testing.assert_allclose(y, expected, rtol=tol, atol=tol)

  @parameterized.parameters(
      (jnp.float32, "add", 1),
      (jnp.float16, "add", 2),
      (jnp.bfloat16, "add", 2),
      (jnp.float16, "min", 4),
      (jnp.float16, "max", 8),
      (jnp.int32, "add", 1),
  )
  def test_reduce_scatter(self, dtype, reduction, vec_size):
    # 16 rows * 64 cols = 1024 elements = 8 elements per thread
    self._test_reduce_scatter(
        (1024, 64), dtype, reduction, tile_size=1024, vec_size=vec_size, num_blocks=4
    )

  def test_reduce_scatter_large_minor_dims(self):
    self._test_reduce_scatter(
        (512, 32768), jnp.float16, "add", tile_size=8192, vec_size=4, num_blocks=4
    )

  @parameterized.parameters(2048, 256, None)
  def test_reduce_scatter_auto_vec_size(self, tile_size):
    self._test_reduce_scatter(
        (1024, 64), jnp.float16, "add", tile_size=tile_size, vec_size=None, num_blocks=4
    )

  @parameterized.parameters(2048, 256, None)
  def test_reduce_scatter_auto_vec_size_int(self, tile_size):
    self._test_reduce_scatter(
        (1024, 64), jnp.int32, "add", tile_size=tile_size, vec_size=None, num_blocks=4
    )

  @parameterized.parameters(1, 2)
  def test_reduce_scatter_different_axes(self, axis):
    if axis == 1:
      shape = (64, 1024, 32)
      tile_size = 2048
    else:  # axis == 2
      shape = (32, 64, 1024)
      tile_size = 2048
    self._test_reduce_scatter(
        shape, jnp.float16, "add", scatter_dimension=axis, tile_size=tile_size, vec_size=None, num_blocks=4
    )

  @parameterized.parameters(
      (jnp.float16, "add"),
      (jnp.float32, "add"),
      (jnp.bfloat16, "max"),
  )
  def test_all_reduce(self, dtype, reduction):
    """Test all-reduce functionality when scatter_dimension=None."""
    self._test_all_reduce(
        (1024, 1024), dtype, reduction, tile_size=512, vec_size=None, num_blocks=4
    )

  def _test_all_reduce(
      self,
      shape,
      dtype,
      reduction,
      tile_size=None,
      vec_size=None,
      num_blocks=None,
  ):
    """Helper function to test all-reduce functionality."""
    self.skip_if_wg_semantics()  # Support multimem_load_reduce under WG.
    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    x = jax.random.normal(jax.random.key(42), (2, *shape), dtype)

    def body(x):
      return reduce_scatter(
          x,
          axis_name="x",
          scatter_dimension=None,  # All-reduce mode
          reduction=reduction,
          vec_size=vec_size,
          tile_size=tile_size,
          num_blocks=num_blocks,
      )

    spec = P("x")
    with mock.patch.object(
        jax.experimental.pallas.ops.gpu.reduce_scatter_mgpu.plgpu,
        "kernel",
        new=self.kernel,
    ):
      y = jax.jit(
          jax.shard_map(
              body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
          )
      )(x)
    y = multihost_utils.process_allgather(y, tiled=True)
    np_reduction = get_reduction_impl(reduction)
    expected = np_reduction(x[0], x[1])
    tol = 1e-5 if reduction == "add" else 0
    for ys in y:
      # It seems that the rounding used by the switch is different from what
      # XLA uses.
      y_rounded = np.nextafter(ys, expected)
      np.testing.assert_allclose(y_rounded, expected, rtol=tol, atol=tol)

  def _test_all_gather(
      self,
      shape,
      dtype,
      gather_dimension=0,
      tile_size=None,
      vec_size=None,
      num_blocks=None,
  ):
    if jax.process_index() > 2:
      return

    if jnp.issubdtype(dtype, jnp.floating):
      x = jax.random.uniform(jax.random.key(42), shape, dtype=dtype, minval=-1.0, maxval=1.0)
    else:
      x = jax.random.randint(jax.random.key(42), shape, dtype=dtype, minval=-1000, maxval=1000)

    def body(x):
      return all_gather(
          x,
          axis_name="x",
          gather_dimension=gather_dimension,
          vec_size=vec_size,
          tile_size=tile_size,
          num_blocks=num_blocks,
      )

    spec = P(*([None] * gather_dimension), "x")
    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ["x"])
    with mock.patch.object(
        jax.experimental.pallas.ops.gpu.all_gather_mgpu.plgpu,
        "kernel",
        self.kernel,
    ):
      y = jax.jit(
          jax.shard_map(
              body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
          )
      )(x)
    y = multihost_utils.process_allgather(y, tiled=True)
    repeats = [1] * len(x.shape)
    repeats[gather_dimension] = 2
    np.testing.assert_array_equal(y, np.tile(x, repeats))

  @parameterized.parameters(
      (jnp.float32, 1),
      (jnp.float16, 2),
      (jnp.bfloat16, 2),
      (jnp.float16, 4),
      (jnp.float16, 8),
      (jnp.int32, 1),
  )
  def test_all_gather(self, dtype, vec_size):
    # 16 rows * 64 cols = 1024 elements = 8 elements per thread
    self._test_all_gather(
        (1024, 64), dtype, tile_size=1024, vec_size=vec_size, num_blocks=4
    )

  def test_all_gather_large_minor_dims(self):
    self._test_all_gather(
        (512, 32768), jnp.float16, tile_size=8192, vec_size=4, num_blocks=4
    )

  @parameterized.parameters(2048, 256, None)
  def test_all_gather_auto_vec_size(self, tile_size):
    self._test_all_gather(
        (1024, 64), jnp.float16, tile_size=tile_size, vec_size=None, num_blocks=4
    )

  @parameterized.parameters(2048, 256, None)
  def test_all_gather_auto_vec_size_int(self, tile_size):
    self._test_all_gather(
        (1024, 64), jnp.int32, tile_size=tile_size, vec_size=None, num_blocks=4
    )

  @parameterized.parameters(1, 2)
  def test_all_gather_different_axes(self, axis):
    if axis == 1:
      shape = (64, 1024, 32)
      tile_size = 2048
    else:  # axis == 2
      shape = (32, 64, 1024)
      tile_size = 2048
    self._test_all_gather(
        shape, jnp.float16, gather_dimension=axis, tile_size=tile_size, vec_size=None, num_blocks=4
    )


class PallasCallRemoteDMAWGTest(
    PallasCallRemoteDMATest,
    lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
):
  ...


class PallasCallMultimemThreadUnsafeWGTest(
    PallasCallMultimemThreadUnsafeTest,
    lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
):
  ...


class PallasCallMultimemWGTest(
    PallasCallMultimemTest,
    lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
):
  ...


if __name__ == '__main__':
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  if is_nvshmem_used():
    # TODO(b/483671897) re-enable once command buffers supported with collectives.
    additional_xla_flags = "--xla_gpu_enable_command_buffer=''"
    if "XLA_FLAGS" in os.environ:
      os.environ["XLA_FLAGS"] = (
          f"{os.environ['XLA_FLAGS']} {additional_xla_flags}"
      )
    else:
      os.environ["XLA_FLAGS"] = additional_xla_flags
    jt_multiprocess.main()
  else:
    config.config_with_absl()
    absltest.main(testLoader=jtu.JaxTestLoader())
