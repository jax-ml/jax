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

import dataclasses

from jax._src import callback
from jax._src import effects
import jax._src.pallas.mosaic.interpret.params as interpret_params

LoggingMode = interpret_params.LoggingMode


@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretGPUParams(interpret_params.SharedInterpretParams):
  """Parameters for GPU interpret mode.

  GPU interpret mode is a way run Pallas GPU kernels on CPU, while simulating
  a GPU's shared memory spaces (GMEM, SMEM, etc.), threads and synchronization
  operations (e.g. barriers). This mode is intended for debugging and testing.

  To run a kernel under GPU interpret mode, pass an instance of
  ``InterpretParams`` as an argument for the ``interpret`` parameter of
  :func:`pallas_call`, :func:`core_map` or :func:`kernel`.

  NOTE: If an exception is raised while interpreting a kernel, you must call
  :func:`reset_gpu_interpret_mode_state` before using GPU interpret mode
  again in the same process.

  Attrs:
    num_tma_threads_per_device: The number of threads that can be used for
      simulating memory transfers with the TMA unit. The interpreter does not
      in fact spawn separate threads for executing TMA memory transfers, but a
      separate vector clock is maintained for each TMA thread.
      Default: 1.
    logging_mode: Logging mode for GPU interpret mode.
  """

  num_tma_threads_per_device: int = 1
  logging_mode: interpret_params.LoggingMode | None = None

  def __post_init__(self):
    super().__post_init__()
    if self.vector_clock_size is not None:
      raise ValueError(
          "`vector_clock_size` must be `None` for GPU interpret mode, but got"
          f" {self.vector_clock_size}."
      )
    if self.num_tma_threads_per_device < 1:
      raise ValueError(
          "Number of TMA threads per device must be at least 1, but got"
          f" {self.num_tma_threads_per_device}."
      )


def get_interpret_effects() -> set[effects.Effect]:
  return {callback._OrderedIOEffect}
