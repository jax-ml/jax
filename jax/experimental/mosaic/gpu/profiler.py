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

import contextlib
import ctypes
import functools
import json

import jax
from jax._src.interpreters import mlir
from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
from jax._src.lib import xla_client
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import scf
import numpy as np

from .utils import *  # noqa: F403

# ruff: noqa: F405
# mypy: ignore-errors

xla_client.register_custom_call_target(
    "mosaic_gpu_record_event",
    mosaic_gpu_lib._mosaic_gpu_ext._record_event_capsule(),
    platform="CUDA",
)

record_event_p = jax.core.Primitive("record_event")
record_event_p.multiple_results = True

@record_event_p.def_abstract_eval
def _record_event_abstract_eval(*args, event):
  del event  # Unused.
  return args

@functools.partial(mlir.register_lowering, record_event_p, platform="cuda")
def _record_event_lowering_rule(ctx, *args, event):
  ptr_bytes = ctypes.cast(event, ctypes.c_void_p).value.to_bytes(
      8, byteorder="little"
  )  # pytype: disable=attribute-error
  op = mlir.custom_call(
      "mosaic_gpu_record_event",
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      operands=args,
      backend_config=ptr_bytes,
      operand_output_aliases={i: i for i in range(len(args))},
  )
  return op.results

def _record_event(args, event):
  flat_args, treedef = jax.tree.flatten(args)
  return jax.tree.unflatten(
      treedef, record_event_p.bind(*flat_args, event=event)
  )

def measure(f, *args):
  # TODO(apaszke): Raise if this is called under jit.
  start_event = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_create()
  end_event = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_create()
  try:
    @jax.jit
    def run(*args):
      return _record_event(f(*_record_event(args, start_event)), end_event)
    results = jax.block_until_ready(run(*args))
    elapsed = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_elapsed(
        start_event, end_event
    )
  finally:
    mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_destroy(start_event)
    mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_destroy(end_event)
  return results, elapsed


class ProfilerSpec:
  ENTER = 0
  EXIT = 1 << 31

  def __init__(self, num_entries: int):
    self.num_entries = num_entries
    self.interned_names = {}

  @property
  def mlir_buffer_type(self) -> ir.Type:
    return ir.MemRefType.get(
        (1 + self.num_entries,), ir.IntegerType.get_signless(32)
    )

  @property
  def jax_buffer_type(self) -> ir.Type:
    return jax.ShapeDtypeStruct((1 + self.num_entries,), jnp.uint32)

  def smem_i32_elements(self, grid: tuple[int, ...]):
    return int(self.num_entries // np.prod(grid))

  def smem_bytes(self, grid: tuple[int, ...]):
    bytes_per_entry = 4
    return self.smem_i32_elements(grid) * bytes_per_entry

  def intern_name(self, name: str) -> int:
    if name_id := self.interned_names.get(name, None):
      return name_id
    name_id = self.interned_names[name] = len(self.interned_names)
    if name_id & self.EXIT:
      raise RuntimeError("Allocated too many names")
    return name_id

  def dump(self, buffer, f):
    buffer = np.asarray(buffer)
    num_blocks = buffer[0]
    per_block = self.num_entries // num_blocks
    block_entries = buffer[1 : 1 + num_blocks * per_block].reshape(
        num_blocks, per_block
    )
    start_times = block_entries[:, :2].astype(np.int64)
    start_times = (start_times[:, 0] << 32) + start_times[:, 1]
    start_times -= start_times.min()  # Normalize
    entries_used = block_entries[:, 2]
    if np.any(entries_used > per_block - 2):
      raise RuntimeError("Insufficient space to capture a full trace")
    block_traces = block_entries[:, 3:]
    unintern = {v: k for k, v in self.interned_names.items()}
    events = []
    for block_idx in range(num_blocks):
      valid_entries = entries_used[block_idx] - 3
      local_clock_offset = None
      assert valid_entries % 2 == 0
      start_time = start_times[block_idx]
      block_events = []
      for i in range(0, valid_entries, 2):
        tag = block_traces[block_idx, i]
        time = block_traces[block_idx, i + 1]
        if local_clock_offset is None:
          local_clock_offset = time
        time -= local_clock_offset
        time -= i * 6  # Account for the overhead of profiling.
        if time < 0:
          break  # Detect a timer wraparound
        name_id = tag
        begin = True
        if name_id & ProfilerSpec.EXIT:
          name_id = name_id ^ ProfilerSpec.EXIT
          begin = False
        name = unintern[name_id]
        block_events.append({
            "name": name,
            "ph": "B" if begin else "E",
            "ts": float(start_time + time) / 1e3,
            "pid": 0,
            "tid": block_idx,
        })
      else:  # If we didn't break
        events.extend(block_events)
    return json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f)


class OnDeviceProfiler:

  def __init__(self, spec: ProfilerSpec, smem_buffer: ir.Value, gmem_buffer: ir.Value):
    self.spec = spec
    # self.should_store = gpu.thread_id(gpu.Dimension.x)
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    num_blocks = c(1, index)
    for dim in gpu.Dimension:
      num_blocks = arith.muli(num_blocks, gpu.grid_dim(dim))
    memref.store(arith.index_cast(i32, num_blocks), gmem_buffer, [c(0, index)])
    self.entries_per_block = arith.divui(c(spec.num_entries, index), num_blocks)
    self.smem_buffer = smem_buffer
    self.gmem_buffer = gmem_buffer
    # Hopefully mem2reg will remove the allocation.
    self.offset = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), self.offset, [])

  @contextlib.contextmanager
  def record(self, name: str):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    name_id = self.spec.intern_name(name)
    def store(modifier):
      cur = arith.index_cast(index, memref.load(self.offset, []))
      # TODO(apaszke): Clamp indices
      # bound = arith.subi(self.entries_per_block, c(2, index))
      # cur = arith.select(
      #     arith.cmpi(arith.CmpIPredicate.ult, cur, bound), cur, bound
      # )
      memref.store(c(modifier | name_id, i32), self.smem_buffer, [cur])
      memref.store(
          clock(), self.smem_buffer, [arith.addi(cur, c(1, cur.type))]
      )
      memref.store(
          arith.index_cast(i32, arith.addi(cur, c(2, cur.type))),
          self.offset,
          [],
      )
    store(ProfilerSpec.ENTER)
    yield
    store(ProfilerSpec.EXIT)

  def finalize(self, grid):
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)

    block_idx = c(0, index)
    for dim in reversed(gpu.Dimension):  # pytype: disable=wrong-arg-types
      block_idx = arith.addi(
          arith.muli(block_idx, gpu.grid_dim(dim)), gpu.block_id(dim)
      )
    start_offset = arith.addi(
        arith.muli(block_idx, self.entries_per_block), c(1, index)
    )
    block_gmem_buffer = memref.subview(
        self.gmem_buffer, [start_offset], [self.spec.num_entries], [1],
        result_type=ir.Type.parse(
            f"memref<{self.spec.num_entries}xi32, strided<[1], offset: ?>>"
        ),
    )
    # TODO(apaszke): Either use globaltimer or delete
    # memref.store(globaltimer("high"), block_gmem_buffer, [c(0, index)])
    # memref.store(globaltimer("low"), block_gmem_buffer, [c(1, index)])
    memref.store(c(0, i32), block_gmem_buffer, [c(0, index)])
    memref.store(c(0, i32), block_gmem_buffer, [c(1, index)])
    memref.store(
        arith.addi(memref.load(self.offset, []), c(3, i32)),
        block_gmem_buffer,
        [c(2, index)],
    )

    if_first = scf.IfOp(
        arith.cmpi(
            arith.CmpIPredicate.eq, gpu.thread_id(gpu.Dimension.x), c(0, index)
        )
    )
    with ir.InsertionPoint(if_first.then_block):
      for_op = scf.ForOp(
          c(0, index),
          c(self.spec.smem_i32_elements(grid) - 3, index),
          c(1, index),
      )
      with ir.InsertionPoint(for_op.body):
        x = memref.load(self.smem_buffer, [for_op.induction_variable])
        memref.store(
            x,
            block_gmem_buffer,
            [arith.addi(for_op.induction_variable, c(3, index))],
        )
        scf.yield_([])
      scf.yield_([])
