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

from collections.abc import Callable
import dataclasses
import enum
from typing import Literal

from jax._src import callback
import numpy as np


class LoggingMode(enum.Flag):
  """Logging mode for the kernel interpreter.

  Attrs:
    BARRIER: Enable logging inside GPU barrier objects.
    SEMAPHORE: Enable logging inside (TPU) semaphore objects.
    SHARED_MEMORY: Enable logging in the shared memory object.
  """

  BARRIER = enum.auto()
  SEMAPHORE = enum.auto()
  SHARED_MEMORY = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True)
class SharedInterpretParams:
  """Parameters for kernel interpret mode.

  Interpret mode is a way to run Pallas kernels on CPU, while simulating TPU/GPU
  shared memory, communication, and synchronization operations.

  Attributes:
    detect_races: If True, a dynamic, happens-before race detector will be used
      to detect data races during kernel interpretation.  If any races are
      detected, a message will be printed and `races.races_found` will be set to
      True.
      Default: False.
    out_of_bounds_reads: If "raise", an exception will be raised on any
      out-of-bounds read of a buffer.  If "uninitialized_value", any parts of
      the read that are out-of-bounds will return the value used to fill
      uninitialized memory, which can be configured via the
      "uninitialized_memory".
      Default: "raise".
    skip_floating_point_ops: If True, operations that produce only floating
      point values will not be interpreted; instead, their results will be
      replaced with arrays all of `jnp.inf`. Additionally any floating point
      operands to any operation will be replaced with (arrays of) `jnp.inf`.
      Default: False.
    uninitialized_memory: If "nan", allocated buffers are initialized to contain
      all NaNs (or to their maximum possible value for integers). If "zero",
      allocated buffers are initialized to all zeros.
      Default: "nan".
    num_cores_or_threads: The number of cores per device (TPU) or threads per
      block (GPU). Note that for interpreting GPU kernels, we currently only
      support a single block in the grid. (So the number of threads per block on
      the GPU can be thought of as the number of threads that runs concurrently
      on the GPU.)
      Default: 1.
    vector_clock_size: The number of entries in the vector clocks. This should
      be an integer bigger then the total number of cores, i.e. bigger than
      `number of devices * num_cores_per_device`. If `None`, the vector clock
      size that is used in the interpreter will default to twice the total
      number of cores.
      This should be left at/set to `None` for interpreting GPU kernels. (For
      GPU kernels, the number of vector clocks is determined by the number of
      devices, `num_cores_or_threads`, and `num_tma_threads_per_device`.)
      Default: None.
    logging_mode: Logging mode for the kernel interpreter.
  """

  detect_races: bool = False
  out_of_bounds_reads: Literal["raise", "uninitialized"] = "raise"
  skip_floating_point_ops: bool = False
  uninitialized_memory: Literal["nan", "zero"] = "nan"
  num_cores_or_threads: int = 1
  vector_clock_size: int | None = None
  logging_mode: LoggingMode | None = None

  def __post_init__(self):
    if self.num_cores_or_threads < 1:
      raise ValueError(
          "Number of cores or threads must be at least 1, but got"
          f" {self.num_cores_or_threads}."
      )
    if self.vector_clock_size is not None and self.vector_clock_size < 1:
      # Further validation is done in `get_vector_clock_size` below.
      raise ValueError(
          "Vector clock size must be at least 1, but got"
          f" {self.vector_clock_size}."
      )

  def get_vector_clock_size(self, num_devices) -> int:
    """Returns the number of vector clocks to use for TPU interpret mode.`"""
    num_cores_or_threads = num_devices * self.num_cores_or_threads
    if self.vector_clock_size is not None:
      if num_cores_or_threads >= self.vector_clock_size:
        raise ValueError(
            f"Vector clock size ({self.vector_clock_size}) must be greater than"
            f" the total number of cores/threads ({num_cores_or_threads})."
        )
      return self.vector_clock_size
    else:
      # Default to twice the total number of cores/threads.
      return 2 * num_cores_or_threads


@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretParams(SharedInterpretParams):
  """Parameters for TPU interpret mode.

  TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating
  a TPU's shared memory (HBM, VMEM, etc.), communication (remote and local
  DMAs), and synchronization operations (semaphores, barriers, etc.).  This mode
  is intended for debugging and testing.

  To run a kernel under TPU interpret mode, pass an instance of
  ``InterpretParams`` as an argument for the ``interpret`` parameter of
  :func:`jax.experimental.pallas.pallas_call` or
  :func:`jax.experimental.pallas.core_map`.

  NOTE: If an exception is raised while interpreting a kernel, you must call
  :func:`reset_tpu_interpret_mode_state` before using TPU interpret mode
  again in the same process.

  Attributes:
    dma_execution_mode:  If "eager", DMAs are executed as soon as they are
      issued.  If "on_wait", DMA reads or writes are only executed when a device
      is waiting on a DMA semaphore that will be signaled when the read or write
      is complete.
      Default: "on_wait".
    random_seed: Seed for random number generator used during interpretation.
      Currently random numbers are used to randomize the grid coordinates along
      dimensions with 'parallel' semantics.
      Default: None.
    grid_point_recorder: Callback that is invoked by the interpreter for each
      grid point in the order in which the grid points are traversed. The
      callback is invoked with two arguments: - A tuple of grid coordinates. -
      The local core ID of the core that is processing the grid point. This
      callback is intended for inspecting - the randomization of coordinates
      along grid dimensions with 'parallel' semantics and - the mapping of grid
      points to local (i.e. per-device) cores.
      Default: None.
    allow_hbm_allocation_in_run_scoped: If `True`, allows the allocation of HBM
      buffers (which are then shared across the cores in a device) in
      `run_scoped`. While this behavior can be enabled in the interpreter,
      allocating HBM buffers with `run_scoped` is not supported when executing
      Pallas kernels on a real TPU.
      Default: `False`.
  """

  dma_execution_mode: Literal["eager", "on_wait"] = "on_wait"
  random_seed: int | None = None
  grid_point_recorder: (
      Callable[[tuple[np.int32, ...], np.int32], None] | None
  ) = None
  allow_hbm_allocation_in_run_scoped: bool = False

  @property
  def num_cores_per_device(self) -> int:
    return self.num_cores_or_threads


def get_interpret_effects():
  return {callback._OrderedIOEffect}
