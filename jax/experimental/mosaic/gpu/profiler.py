# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Callable
import contextlib
import itertools
import json
import math
import os
import tempfile
from typing import Literal, ParamSpec, TypeVar, overload
import warnings

import jax
from jax._src import stages
from jax._src import util
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
import numpy as np

from .utils import *  # noqa: F403

try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
except ImportError:
  mosaic_gpu_lib = None  # type: ignore[assignment]

# ruff: noqa: F405

T = TypeVar("T")
P = ParamSpec("P")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cupti:
  """CUPTI-based profiler."""

  # If `True`, detach CUPTI from the process after measurement.
  finalize: bool = True

  def measure(
      self, f, *, aggregate: bool = True, iterations: int = 1,
  ):
    if not isinstance(f, (stages.Wrapped, stages.Compiled)):
      f = jax.jit(f)

    def wrapper(*args, **kwargs):
      if mosaic_gpu_lib is None:
        raise RuntimeError("CUPTI profiling is not supported on this platform")

      jax.block_until_ready(f(*args, **kwargs))  # Warmup.
      ext = mosaic_gpu_lib._mosaic_gpu_ext
      ext._cupti_init()
      try:
        all_results = [f(*args, **kwargs) for _ in range(iterations)]
        for r in all_results:
          jax.block_until_ready(r)
        results = all_results[0]
      finally:
        timings = ext._cupti_get_timings(self.finalize)
      if not timings:
        return results, None

      if len(timings) % iterations != 0:
        raise RuntimeError(
            "The number of kernel launches is not divisible by the number of"
            " iterations"
        )
      kernels_per_iter = len(timings) // iterations
      iter_timings = util.split_list(
          timings, [kernels_per_iter] * (iterations - 1)
      )
      for kernel_idx, (kernel_name, _) in enumerate(iter_timings[0]):
        for i in range(1, iterations):
          if iter_timings[i][kernel_idx][0] != kernel_name:
            raise RuntimeError("Kernel names are not consistent across iterations")

      if aggregate:
        iter_timings = [
            sum(item[1] for item in timings) for timings in iter_timings
        ]

      return results, iter_timings[0] if len(iter_timings) == 1 else iter_timings

    return wrapper

@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[True] = ...,
    iterations: Literal[1] = ...,
) -> Callable[P, tuple[T, float | None]]:
  ...

@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[False] = ...,
    iterations: Literal[1] = ...,
) -> Callable[P, tuple[T, list[tuple[str, float]] | None]]:
  ...

@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[True] = ...,
    iterations: int = ...,
) -> Callable[P, tuple[T, list[float] | None]]:
  ...

@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[False] = ...,
    iterations: int = ...,
) -> Callable[P, tuple[T, list[list[tuple[str, float]]] | None]]:
  ...


def measure(
    f, *, aggregate: bool = True, iterations: int = 1,
):
  """Measures the GPU runtime of a function using CUPTI.

  ``measure`` is a higher-order function that wraps a function ``f`` to
  return GPU runtime in milliseconds, in addition to its regular outputs.

  Args:
    f: The function to measure.
    aggregate: Whether to report an aggregate runtime. When ``False`` (only
      supported by ``mode="cupti"``), the per-kernel timings are returned as a
      list of tuples ``(<kernel name>, <runtime in ms>)``.
    iterations: How many times to run the function. Only supported by
      ``mode="cupti"``. When greater than 1, the return type will become a list
      of measurements.

  Returns:
    A function that accepts the same inputs as ``f`` and returns
    ``(f_outputs, timings)``, where ``f_outputs`` are the outputs of ``f``,
    and ``timings`` is either a float or a list of tuples, depending on
    ``aggregate``. If no kernels are launched, ``timings`` is ``None``.

  Notes:
    `CUPTI (CUDA Profiling Tools Interface)
    <https://docs.nvidia.com/cupti/index.html>`_ is a high-accuracy profiling
    API used by Nsight Systems and Nsight Compute. The CUPTI API only allows a
    single subscriber, so ``measure`` cannot be used with other CUPTI-based
    tools like CUDA-GDB, Compute Sanitizer, Nsight Systems, or Nsight
    Compute.
  """  # fmt: skip
  if iterations < 1:
    raise ValueError(f"{iterations=} must be positive")
  return Cupti().measure(f, aggregate=aggregate, iterations=iterations)


class ProfilerSpec:
  ENTER = 0
  EXIT = 1 << 31

  def __init__(self, entries_per_warpgroup: int, dump_path: str = "sponge"):
    self.entries_per_warpgroup = entries_per_warpgroup
    self.interned_names: dict[str, int] = {}
    if dump_path == "sponge":
      self.dump_path = os.getenv(
          "TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir()
      )
    else:
      self.dump_path = dump_path

  def _num_warpgroups(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> int:
    if math.prod(block) % WARPGROUP_SIZE:
      raise ValueError("Block size is not a multiple of warpgroup size")
    return math.prod(grid) * math.prod(block) // WARPGROUP_SIZE

  def mlir_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return ir.MemRefType.get(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        ir.IntegerType.get_signless(32),
    )

  def jax_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return jax.ShapeDtypeStruct(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        jnp.uint32,
    )

  def smem_i32_elements(self, block: tuple[int, ...]):
    num_warpgroups = self._num_warpgroups((), block)
    return int(num_warpgroups * self.entries_per_warpgroup)

  def smem_bytes(self, block: tuple[int, ...]):
    bytes_per_entry = 4
    return self.smem_i32_elements(block) * bytes_per_entry

  def intern_name(self, name: str) -> int:
    if (name_id := self.interned_names.get(name, None)) is not None:
      return name_id
    name_id = self.interned_names[name] = len(self.interned_names)
    if name_id & self.EXIT:
      raise RuntimeError("Allocated too many names")
    return name_id

  def dump(self, buffer, f, grid: tuple[int, ...], block: tuple[int, ...]):
    buffer = np.asarray(buffer)
    num_blocks = math.prod(grid)
    warpgroups_per_block = self._num_warpgroups((), block)
    entries = buffer.reshape(
        num_blocks, warpgroups_per_block, self.entries_per_warpgroup
    )
    start_times = entries[..., 0]
    sm_ids = entries[..., 1]
    traces_used = entries[..., 2]
    entries_used = traces_used + 3
    if np.any(entries_used > self.entries_per_warpgroup):
      raise RuntimeError("Insufficient space to capture a full trace")
    traces = entries[..., 3:]

    # Estimate the overhead of profiling.
    time_events = traces[:, :, 1::2]
    valid_times_mask = np.arange(traces.shape[-1])[1::2] < traces_used[..., None]
    # 12 cycles is a ballpark estimate for H100
    profiling_overhead = (time_events[:, :, 1:] - time_events[:, :, :-1]).min(
        where=valid_times_mask[:, :, 1:], initial=12
    )
    profiling_overhead = max(0, profiling_overhead - 1)

    unintern = {v: k for k, v in self.interned_names.items()}
    events = []
    for block_idx, wg_idx in np.ndindex(num_blocks, warpgroups_per_block):
      valid_entries = traces_used[block_idx, wg_idx]
      local_clock_offset = None
      assert valid_entries % 2 == 0, valid_entries
      start_time = start_times[block_idx, wg_idx]
      block_events = []
      last_time = float("-inf")
      for i in range(0, valid_entries, 2):
        tag = traces[block_idx, wg_idx, i]
        time = traces[block_idx, wg_idx, i + 1]
        if local_clock_offset is None:
          local_clock_offset = time
        time -= local_clock_offset
        time -= (i // 2) * profiling_overhead  # Account for the overhead of profiling.
        if time < 0:
          break  # Detect a timer wraparound
        name_id = tag
        begin = True
        if name_id & ProfilerSpec.EXIT:
          name_id = name_id ^ ProfilerSpec.EXIT
          begin = False
        name = unintern[name_id]
        if last_time >= time:
          if last_time - time > 10:
            warnings.warn(
                "Profiler clock went significantly backwards for event"
                f" {'start' if begin else 'end'} `{name}`: {last_time} ->"
                f" {time}"
            )
          time = last_time + 1
        last_time = time
        block_events.append({
            "name": name,
            "ph": "B" if begin else "E",
            "ts": float(start_time + time) / 1e3,
            "pid": 1 + int(sm_ids[block_idx, wg_idx]),
            "tid": 1 + wg_idx + warpgroups_per_block * block_idx,
        })
      else:  # If we didn't break
        if block_events:
          events.append(block_events)
    events = sorted(events, key=lambda x: x[0]["ts"])
    flat_events = list(itertools.chain.from_iterable(events))
    return json.dump({"displayTimeUnit": "ns", "traceEvents": flat_events}, f)


@dataclasses.dataclass(frozen=True)
class _ProfilerCtx:
  """Set of IR values referenced by the profiler logic.

  The profiler logic is implemented using `CustomPrimitiveOp` which requires
  that all IR values referenced in its body be passed as operands to the op.
  """

  start: ir.Value
  is_profiling_thread: ir.Value
  smem_buffer: ir.Value
  gmem_buffer: ir.Value
  offset: ir.Value


class OnDeviceProfiler:

  def __init__(
      self,
      spec: ProfilerSpec,
      smem_buffer: ir.Value,
      gmem_buffer: ir.Value,
      wrap_in_custom_primitive: bool,
  ):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    self.spec = spec
    self.entries_per_wg = spec.entries_per_warpgroup
    self.wrap_in_custom_primitive = wrap_in_custom_primitive
    wg_idx = warpgroup_idx(sync=False)
    wg_offset = arith.index_cast(
        index, arith.muli(wg_idx, c(self.entries_per_wg, i32))
    )
    smem_buffer = memref_slice(smem_buffer, ds(wg_offset, self.entries_per_wg))
    is_profiling_thread = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.remui(thread_idx(), c(WARPGROUP_SIZE, i32)),
        c(0, i32),
    )
    # Hopefully mem2reg will remove the allocation.
    offset = memref.alloca(ir.MemRefType.get((), index), [], [])
    memref.store(c(0, index), offset, [])
    self.ctx = _ProfilerCtx(
        start=globaltimer("low"),
        is_profiling_thread=is_profiling_thread,
        smem_buffer=smem_buffer,
        gmem_buffer=gmem_buffer,
        offset=offset,
    )

  @contextlib.contextmanager
  def _profiler_ctx(self):
    if not self.wrap_in_custom_primitive:
      yield self.ctx
      return

    def fields(obj) -> list[ir.Value]:
      return [getattr(obj, field.name) for field in dataclasses.fields(obj)]

    op = dialect.CustomPrimitiveOp(
        result=[],
        operands_=fields(self.ctx),
        in_layouts=[],
        in_transforms=[ir.ArrayAttr.get([])],
        out_layouts=[],
    )
    args_ty = [arg.type for arg in op.operands_]
    block = op.body.blocks.append(*args_ty)
    with ir.InsertionPoint(block):
      yield _ProfilerCtx(*block.arguments)
      dialect.return_([])

  @contextlib.contextmanager
  def record(self, name: str):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    name_id = self.spec.intern_name(name)
    def store(modifier):
      with self._profiler_ctx() as ctx:
        # smem_buffer[offset] = modifier | name_id
        # smem_buffer[offset + 1] = %clock
        # offset += 2
        offset = memref.load(ctx.offset, [])
        base_ref = memref_slice(ctx.smem_buffer, offset)
        base_ptr = memref_ptr(base_ref, memory_space=3)
        i64 = ir.IntegerType.get_signless(64)
        base_addr = llvm.ptrtoint(i64, base_ptr)
        llvm.inline_asm(
            ir.Type.parse("!llvm.void"),
            [ctx.is_profiling_thread, base_addr, c(modifier | name_id, i32)],
            """
            @$0 st.shared.v2.u32 [$1], {$2, %clock};
            """,
            "b,l,r",
            has_side_effects=True,
        )
        new_offset = arith.addi(offset, c(2, index))
        memref.store(new_offset, ctx.offset, [])

    store(ProfilerSpec.ENTER)
    yield
    store(ProfilerSpec.EXIT)

  def finalize(self, grid: tuple[int, ...], block: tuple[int, ...]):
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)

    with self._profiler_ctx() as ctx:
      gpu.barrier()  # Make sure all warpgroups are done.

      block_idx = c(0, index)
      for dim in gpu.Dimension:  # pytype: disable=wrong-arg-types
        block_idx = arith.addi(
            arith.muli(block_idx, gpu.grid_dim(dim)), gpu.block_id(dim)
        )
      wg_idx = warpgroup_idx(sync=False)
      wg_per_block = math.prod(block) // WARPGROUP_SIZE
      global_wg_idx = arith.addi(
          arith.muli(block_idx, c(wg_per_block, index)),
          arith.index_cast(index, wg_idx),
      )
      start_offset = arith.muli(global_wg_idx, c(self.entries_per_wg, index))
      wg_gmem_buffer = memref_slice(
          ctx.gmem_buffer, ds(start_offset, self.entries_per_wg)
      )
      with when(ctx.is_profiling_thread):
        memref.store(ctx.start, wg_gmem_buffer, [c(0, index)])
        memref.store(smid(), wg_gmem_buffer, [c(1, index)])
        num_traces = arith.index_cast(i32, memref.load(ctx.offset, []))
        memref.store(num_traces, wg_gmem_buffer, [c(2, index)])
        traces = vector.load(
            ir.VectorType.get((self.entries_per_wg - 3,), i32),
            ctx.smem_buffer,
            [c(0, index)],
        )
        vector.store(traces, wg_gmem_buffer, [c(3, index)])
