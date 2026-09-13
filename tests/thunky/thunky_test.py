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

"""Tests for the JAX Jaxpr embedding and Jaxpr-to-MLIR translator of `thunky`."""

import inspect
import os
import re
import textwrap

from absl.testing import absltest
import jax
from jax import shard_map
from jax._src import config
from jax._src import test_util as jtu
from jax._src.state import discharge as state_discharge
from jax._src.thunky.thunky import (
    ReadEffect,
    WriteEffect,
    jax_executable_to_mlir,
    jax_executable_to_mlir_text,
    make_buffer_aval,
    make_ir_context,
)
from jax.experimental import thunky
from jax.experimental.mosaic import gpu as mgpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


def _normalize_mlir(mlir_text: str) -> str:
  text = re.sub(
      r'(asm_text|binary|ptx|thunk_proto|kernel_hash|module|executable_metadata_proto)\s*=\s*"[^"]*"',
      r'\1 = "..."',
      mlir_text,
  )
  text = re.sub(r'name\s*=\s*"[^"]*"', 'name = "..."', text)
  return text.strip()


ADD_ONE_PTX = """
.version 7.0
.target sm_80
.address_size 64

.visible .entry add_one_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr
)
{
    .reg .u64 %rd<3>;
    .reg .u32 %r1;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];

    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd0, %r1, 4;
    add.u64 %rd1, %rd1, %rd0;
    add.u64 %rd2, %rd2, %rd0;

    ld.global.f32 %f0, [%rd1];
    add.f32 %f1, %f0, 0f3F800000;
    st.global.f32 [%rd2], %f1;
    ret;
}
"""


class ThunkyJaxTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("ThunkyJaxTest requires a GPU backend")

  # ============================================================================
  # Core Primitives & Scratch Buffers
  # ============================================================================

  def test_basic_primitives_and_scratch(self):
    """Traces and lowers a thunky Jaxpr program using primitive wrappers."""

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((4,), jnp.float32))
    def prog(in_buf, out_buf, tmp):
      thunky.memzero(tmp)
      thunky.copy(in_buf, tmp)
      thunky.ptx_kernel(
          tmp,
          out_buf,
          written=[False, True],
          kernel_name="add_one_kernel",
          ptx=ADD_ONE_PTX,
          grid_dim=(1, 1, 1),
          block_dim=(4, 1, 1),
      )

    x = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    out = jnp.zeros_like(x)

    closed_jaxpr = prog.trace_jaxpr(x, out)
    prim_names = [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns]
    self.assertEqual(
        prim_names,
        [
            "thunky.memzero",
            "thunky.copy",
            "thunky.ptx_kernel",
        ],
    )

    mlir_mod = prog.lower_to_mlir(x, out)
    self.assertEqual(
        _normalize_mlir(str(mlir_mod)),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<16>, %arg1: !thunky.buffer<16>, %arg2: !thunky.buffer<16>) {
                thunky.memzero %arg2 : <16>
                thunky.copy %arg0, %arg2 : <16>, <16>
                thunky.ptx_kernel %arg2, %arg1 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {block_dim = array<i64: 4, 1, 1>, grid_dim = array<i64: 1, 1, 1>, ptx = "...", shmem_bytes = 0 : i64, written = array<i1: false, true>}
                return
              }
            }"""),
    )

    out_ref = jax.new_ref(out)
    prog(x, out_ref)
    np.testing.assert_allclose(
        np.asarray(out_ref[...]),
        np.array([11.0, 21.0, 31.0, 41.0], dtype=np.float32),
    )

  def test_call_jax_splicing(self):
    """Verifies thunky.call_jax compiles and splices jitted JAX functions."""

    def scale_and_bias(a_ref, b_ref, out_ref):
      out_ref[...] = a_ref[...] * 2.0 + b_ref[...]

    def square_and_sub(a_ref, b_ref, out_ref):
      out_ref[...] = a_ref[...] * a_ref[...] - b_ref[...]

    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        )
    )
    def pipeline(x_buf, y_buf, out_buf, const_buf, mid_buf):
      thunky.memset(const_buf, np.float32(3.0))
      thunky.call_jax(scale_and_bias, x_buf, const_buf, mid_buf)
      thunky.call_jax(square_and_sub, mid_buf, y_buf, out_buf)

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    y = jnp.array([0.5, 1.5, 2.5, 3.5], dtype=jnp.float32)
    out = jnp.zeros_like(x)

    closed_jaxpr = pipeline.trace_jaxpr(x, y, out)
    prim_names = [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns]
    self.assertEqual(
        prim_names,
        [
            "thunky.memset",
            "thunky.call_jax",
            "thunky.call_jax",
        ],
    )

    self.assertEqual(
        _normalize_mlir(str(pipeline.lower_to_mlir(x, y, out))),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<16>, %arg1: !thunky.buffer<16>, %arg2: !thunky.buffer<16>, %arg3: !thunky.buffer<16>, %arg4: !thunky.buffer<16>, %arg5: !thunky.buffer<24>, %arg6: !thunky.buffer<24>) {
                thunky.memset32 %arg3, 1077936128 : <16>
                thunky.call_thunk_proto %arg3, %arg0, %arg4 : !thunky.buffer<16>, !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                thunky.call_thunk_proto %arg1, %arg4, %arg2 : !thunky.buffer<16>, !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                return
              }
            }"""),
    )

    out_ref = jax.new_ref(out)
    pipeline(x, y, out_ref)
    expected = (np.asarray(x) * 2.0 + 3.0) ** 2 - np.asarray(y)
    np.testing.assert_allclose(np.asarray(out_ref[...]), expected, rtol=1e-5)

  def test_call_jax_in_place_mutation(self):
    """Verifies in-place Ref read-modify-write via call_jax elides self-copies and infers ReadEffect/WriteEffect."""

    def accum_kernel(x_ref, y_ref):
      y_ref[...] += x_ref[...] * 2.0

    @thunky.jit
    def accum_thunky(x_buf, y_buf):
      thunky.call_jax(accum_kernel, x_buf, y_buf)

    closed_jaxpr = accum_thunky.trace_jaxpr(
        jax.ShapeDtypeStruct((8,), jnp.float32),
        jax.ShapeDtypeStruct((8,), jnp.float32),
    )
    x_var, y_var = closed_jaxpr.jaxpr.invars
    self.assertIn(ReadEffect(x_var), closed_jaxpr.jaxpr.effects)
    self.assertNotIn(WriteEffect(x_var), closed_jaxpr.jaxpr.effects)
    self.assertIn(ReadEffect(y_var), closed_jaxpr.jaxpr.effects)
    self.assertIn(WriteEffect(y_var), closed_jaxpr.jaxpr.effects)

    mlir_mod = accum_thunky.lower_to_mlir(
        jax.ShapeDtypeStruct((8,), jnp.float32),
        jax.ShapeDtypeStruct((8,), jnp.float32),
    )
    self.assertNotIn("thunky.copy", str(mlir_mod))

    x = jnp.arange(8, dtype=jnp.float32)
    y = jnp.ones(8, dtype=jnp.float32) * 10.0
    y_ref = jax.new_ref(y)
    accum_thunky(x, y_ref)
    np.testing.assert_allclose(
        np.asarray(y_ref[...]),
        np.ones(8, dtype=np.float32) * 10.0 + np.asarray(x) * 2.0,
        rtol=1e-5,
    )

  def test_call_jax_aliasing_value_inputs_and_returned_refs(self):
    """Verifies call_jax supports in-place Ref aliasing, read-only value inputs, and returned new Refs."""
    expected_const = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    def _set_const(r):
      r[...] = jnp.array(expected_const)

    @thunky.jit
    def set_constant_in_place(x_ref):
      thunky.call_jax(_set_const, x_ref)

    @jax.jit
    def run_set_constant():
      r = jax.new_ref(jnp.zeros((4,), dtype=jnp.float32))
      set_constant_in_place(r)
      return r[...]

    np.testing.assert_allclose(np.asarray(run_set_constant()), expected_const)

    # In-place GEMM (`r[...] = r[...] @ r[...]`) and Ref slices (`x_ref.at[:]`)
    def _square_mat(r):
      r[...] = r[...] @ r[...]

    @thunky.jit
    def square_in_place(mat_ref):
      thunky.call_jax(_square_mat, mat_ref.at[:])

    @jax.jit
    def run_square(m):
      r = jax.new_ref(m)
      square_in_place(r)
      return r[...]

    m_np = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(
        np.asarray(run_square(jnp.array(m_np))), m_np @ m_np, rtol=1e-5
    )

    # Value inputs (`x_ref[:]`) passed to `fn` as read-only `jax.Array`
    @thunky.jit
    def value_in_ref_out(x_ref, y_ref):
      def _scale(x_val, y_r):
        self.assertNotIsInstance(x_val, jax._src.state.types.AbstractRef)
        y_r[...] = x_val * 3.0 + 1.0

      thunky.call_jax(_scale, x_ref[:], y_ref.at[:])

    @jax.jit
    def run_value_in_ref_out(x):
      rx = jax.new_ref(x)
      ry = jax.new_ref(jnp.zeros_like(x))
      value_in_ref_out(rx, ry)
      return ry[...]

    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(run_value_in_ref_out(jnp.array(x_np))), x_np * 3.0 + 1.0
    )

    # `fn` returning a value (or pytree of values) creates new Refs in the enclosing program
    def _add_five(r):
      r[...] = r[...] + 5.0

    @thunky.jit
    def matmul_returning_new_ref(a_ref, b_ref, out_ref):
      c_ref = thunky.call_jax(lambda a, b: a @ b, a_ref[:], b_ref[:])
      thunky.call_jax(_add_five, c_ref)
      thunky.copy(c_ref, out_ref)

    @jax.jit
    def run_matmul_new_ref(a, b):
      ra = jax.new_ref(a)
      rb = jax.new_ref(b)
      rout = jax.new_ref(jnp.zeros_like(a))
      matmul_returning_new_ref(ra, rb, rout)
      return rout[...]

    a_np = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    b_np = np.eye(4, dtype=np.float32) * 2.0
    np.testing.assert_allclose(
        np.asarray(run_matmul_new_ref(jnp.array(a_np), jnp.array(b_np))),
        (a_np @ b_np) + 5.0,
        rtol=1e-5,
    )

  def test_closed_over_constants_in_call_jax_and_thunky_jit(self):
    """Verifies closed-over array constants in call_jax and @thunky.jit preserve parameter effect indices and lower cleanly."""
    c = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    def _add_const(x, y):
      y[...] = x[...] + c

    @thunky.jit
    def prog_with_call_jax_const(x_ref, y_ref):
      thunky.call_jax(_add_const, x_ref, y_ref)

    aval = make_buffer_aval((4,), jnp.float32)
    closed_jaxpr = prog_with_call_jax_const.trace_jaxpr(aval, aval)
    write_indices = {
        eff.input
        if isinstance(eff.input, int)
        else closed_jaxpr.jaxpr.invars.index(eff.input)
        for eff in closed_jaxpr.jaxpr.effects
        if isinstance(eff, jax._src.state.types.WriteEffect)
    }
    self.assertIn(1, write_indices)
    self.assertNotIn(0, write_indices)

    @thunky.jit
    def prog_with_direct_const(x_ref, y_ref):
      y_ref[...] = x_ref[...] + c

    prog_with_direct_const.lower_to_mlir(aval, aval)

    @jax.jit
    def run_direct_const(x):
      rx = jax.new_ref(x)
      ry = jax.new_ref(jnp.zeros_like(x))
      prog_with_direct_const(rx, ry)
      return ry[...]

    x_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    self.assertArraysAllClose(
        np.asarray(run_direct_const(jnp.array(x_np))),
        x_np + np.asarray(c),
    )

  def test_eager_execution_rejects_mutating_immutable_jax_array(self):
    """Verifies calling a mutating @thunky.jit function eagerly with an immutable jax.Array raises TypeError."""

    @thunky.jit
    def zero_buffer(x_ref):
      thunky.memzero(x_ref)

    x = jnp.ones((4,), dtype=jnp.float32)
    with self.assertRaises(TypeError):
      zero_buffer(x)

  # ============================================================================
  # Buffer Views, Slicing & Direct `Ref` Indexing
  # ============================================================================

  def test_buffer_slicing_and_ref_indexing(self):
    """Verifies zero-copy buffer slicing via Ref.at[...] views and Ref[...] = ...

    assignment.
    """

    def add_one_ref(x_ref, out_ref):
      out_ref[...] = x_ref[...] + 1.0

    def accum_kernel(x_ref, y_ref):
      y_ref[...] += x_ref[...] * 2.0

    @jax.jit
    def add_one_val(x):
      return x + 1.0

    @thunky.jit
    def sliced_writes_fn(in_buf, out_buf):
      # Slice input and output buffers along axis 0 using Ref.at[...] views
      in_top = in_buf.at[:4]
      in_bot = in_buf.at[4:]
      out_top = out_buf.at[:4]
      out_bot = out_buf.at[4:]

      # Write to top half via call_jax, copy bottom half directly
      thunky.call_jax(add_one_ref, in_top, out_top)
      thunky.copy(in_bot, out_bot)

    @thunky.jit
    def sliced_accum_fn(x_buf, y_buf):
      thunky.call_jax(accum_kernel, x_buf.at[:4], y_buf.at[:4])
      thunky.call_jax(accum_kernel, x_buf.at[4:], y_buf.at[4:])

    @thunky.jit
    def ref_assignment_fn(in_buf, out_buf):
      # Direct Ref indexing and assignment lowered via get_p / swap_p
      out_buf[:4] = add_one_val(in_buf[:4])
      out_buf[4:] = in_buf[4:]

    x_np = np.arange(32, dtype=np.float32).reshape((8, 4))
    expected = np.concatenate([x_np[:4] + 1.0, x_np[4:]], axis=0)

    x = jnp.asarray(x_np)
    out1 = jnp.zeros((8, 4), dtype=jnp.float32)

    self.assertEqual(
        _normalize_mlir(str(sliced_writes_fn.lower_to_mlir(x, out1))),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<128>, %arg1: !thunky.buffer<128>, %arg2: !thunky.buffer<16>) {
                %0 = thunky.slice_buffer %arg0 offset = 0 : <128> -> <64>
                %1 = thunky.slice_buffer %arg1 offset = 0 : <128> -> <64>
                thunky.call_thunk_proto %0, %1 : !thunky.buffer<64>, !thunky.buffer<64> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                %2 = thunky.slice_buffer %arg0 offset = 64 : <128> -> <64>
                %3 = thunky.slice_buffer %arg1 offset = 64 : <128> -> <64>
                thunky.copy %2, %3 : <64>, <64>
                return
              }
            }"""),
    )

    out1_ref = jax.new_ref(out1)
    sliced_writes_fn(x, out1_ref)
    np.testing.assert_allclose(np.asarray(out1_ref[...]), expected, rtol=1e-5)

    out2 = jnp.zeros((8, 4), dtype=jnp.float32)
    out2_ref = jax.new_ref(out2)
    ref_assignment_fn(x, out2_ref)
    np.testing.assert_allclose(np.asarray(out2_ref[...]), expected, rtol=1e-5)

    x_1d = jnp.arange(8, dtype=jnp.float32)
    y_1d = jnp.ones(8, dtype=jnp.float32) * 10.0
    y_1d_ref = jax.new_ref(y_1d)
    sliced_accum_fn(x_1d, y_1d_ref)
    np.testing.assert_allclose(
        np.asarray(y_1d_ref[...]),
        np.ones(8, dtype=np.float32) * 10.0 + np.asarray(x_1d) * 2.0,
        rtol=1e-5,
    )

    # Verify Jaxpr effects record ReadEffect and WriteEffect on base buffers
    closed_jaxpr = sliced_writes_fn.trace_jaxpr(
        jax.ShapeDtypeStruct((8, 4), jnp.float32),
        jax.ShapeDtypeStruct((8, 4), jnp.float32),
    )
    in_var, out_var = closed_jaxpr.jaxpr.invars
    self.assertIn(ReadEffect(in_var), closed_jaxpr.jaxpr.effects)
    self.assertIn(WriteEffect(out_var), closed_jaxpr.jaxpr.effects)

  def test_ref_swap_war_hazard_materialization(self):
    """Verifies lazy Ref reads materialize scratch copies when overwritten before subsequent reads."""

    @thunky.jit
    def swap_buffers(a_ref, b_ref):
      a = a_ref[...]
      b = b_ref[...]
      a_ref[...] = b
      b_ref[...] = a

    aval = make_buffer_aval((4,), jnp.float32)
    module = swap_buffers.lower_to_mlir(aval, aval)
    main_func = [
        op.operation
        for op in module.body.operations
        if op.operation.name == "func.func"
    ][0]
    block = main_func.regions[0].blocks[0]
    arg0, arg1 = block.arguments[0], block.arguments[1]

    copy_ops = [
        op.operation
        for op in block.operations
        if op.operation.name == "thunky.copy"
    ]
    arg0_overwritten = False
    war_hazard_detected = False
    for cop in copy_ops:
      src, dst = cop.operands[0], cop.operands[1]
      if dst == arg0 and src == arg1:
        arg0_overwritten = True
      elif arg0_overwritten and src == arg0 and dst == arg1:
        war_hazard_detected = True

    self.assertFalse(war_hazard_detected)

    @jax.jit
    def run_swap(a, b):
      ra = jax.new_ref(a)
      rb = jax.new_ref(b)
      swap_buffers(ra, rb)
      return ra[...], rb[...]

    a_val = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    b_val = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    res_a, res_b = run_swap(a_val, b_val)
    np.testing.assert_allclose(np.asarray(res_a), np.asarray(b_val), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(res_b), np.asarray(a_val), rtol=1e-5)

  def test_buffer_slice_bounds_and_negative_start_index(self):
    """Verifies partial trailing dimension slices raise ValueError and negative start indices normalize cleanly."""

    @thunky.jit
    def partial_trailing_slice(buf_ref):
      thunky.memzero(buf_ref.at[:2, :4])

    aval = make_buffer_aval((4, 8), jnp.float32)
    with self.assertRaises(ValueError):
      partial_trailing_slice.lower_to_mlir(aval)

    @thunky.jit
    def negative_start_slice(buf_ref):
      thunky.memzero(buf_ref.at[-2:])

    module = negative_start_slice.lower_to_mlir(aval)
    main_func = [
        op.operation
        for op in module.body.operations
        if op.operation.name == "func.func"
    ][0]
    slice_ops = [
        op.operation
        for op in main_func.regions[0].blocks[0].operations
        if op.operation.name == "thunky.slice_buffer"
    ]
    self.assertNotEmpty(slice_ops)
    offset_val = int(slice_ops[0].attributes["offset"])
    self.assertEqual(offset_val, 64)

  # ============================================================================
  # Composition (Nested `@thunky.jit`)
  # ============================================================================

  def test_nested_thunky_jit(self):
    """Verifies calling one @thunky.jit function inside another, effect propagation, and scratch buffer hoisting."""

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((4,), jnp.float32))
    def inner_thunky(src_ref, dst_ref, inner_scratch_ref):
      def scale_and_shift(x_ref, s_ref):
        s_ref[...] = x_ref[...] * 2.0 + 1.0

      def add_to_dst(s_ref, d_ref):
        d_ref[...] = d_ref[...] + s_ref[...]

      thunky.call_jax(scale_and_shift, src_ref, inner_scratch_ref)
      thunky.call_jax(add_to_dst, inner_scratch_ref, dst_ref)

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((8,), jnp.float32))
    def outer_thunky(x_ref, out_ref, outer_scratch_ref):
      thunky.memzero(outer_scratch_ref)
      inner_thunky(x_ref.at[0:4], outer_scratch_ref.at[0:4])
      inner_thunky(x_ref.at[4:8], outer_scratch_ref.at[4:8])
      thunky.copy(outer_scratch_ref, out_ref)

    closed_jaxpr = outer_thunky.trace_jaxpr(
        jax.ShapeDtypeStruct((8,), jnp.float32),
        jax.ShapeDtypeStruct((8,), jnp.float32),
    )
    nested_eqns = [
        eqn
        for eqn in closed_jaxpr.jaxpr.eqns
        if eqn.primitive is thunky.call_thunky_p
    ]
    self.assertLen(nested_eqns, 2)
    for eqn in nested_eqns:
      self.assertIn(jax._src.state.types.ReadEffect(eqn.invars[0]), eqn.effects)
      self.assertIn(
          jax._src.state.types.WriteEffect(eqn.invars[1]), eqn.effects
      )

    x_np = np.arange(8, dtype=np.float32)
    expected = x_np * 2.0 + 1.0

    x = jnp.asarray(x_np)
    out = jnp.zeros((8,), dtype=jnp.float32)
    self.assertEqual(
        _normalize_mlir(str(outer_thunky.lower_to_mlir(x, out))),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<32>, %arg1: !thunky.buffer<32>, %arg2: !thunky.buffer<32>, %arg3: !thunky.buffer<16>, %arg4: !thunky.buffer<16>, %arg5: !thunky.buffer<16>, %arg6: !thunky.buffer<16>, %arg7: !thunky.buffer<16>, %arg8: !thunky.buffer<16>) {
                thunky.memzero %arg2 : <32>
                %0 = thunky.slice_buffer %arg0 offset = 0 : <32> -> <16>
                %1 = thunky.slice_buffer %arg2 offset = 0 : <32> -> <16>
                thunky.call_thunk_proto %0, %arg3 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                thunky.call_thunk_proto %1, %arg3 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                %2 = thunky.slice_buffer %arg0 offset = 16 : <32> -> <16>
                %3 = thunky.slice_buffer %arg2 offset = 16 : <32> -> <16>
                thunky.call_thunk_proto %2, %arg6 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                thunky.call_thunk_proto %3, %arg6 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                thunky.copy %arg2, %arg1 : <32>, <32>
                return
              }
            }"""),
    )
    out_ref = jax.new_ref(out)
    outer_thunky(x, out_ref)
    np.testing.assert_allclose(np.asarray(out_ref[...]), expected, rtol=1e-5)

  def test_nested_embedded_thunk_preserves_kernel_binary_and_constant_offsets(
      self,
  ):
    """Verifies EmbeddedThunkWrapper::ToProto() preserves CUBIN/PTX binaries and internal constant offsets across nested call_jax splicing."""
    const_vec = jnp.arange(16, dtype=jnp.float32) + 10.0

    def _add_const(r):
      r[...] = r[...] + const_vec

    @thunky.jit
    def inner_thunky(x_ref):
      thunky.call_jax(_add_const, x_ref)

    def mid_jax(x_ref):
      inner_thunky(x_ref)

    @thunky.jit
    def outer_thunky(x_ref):
      thunky.call_jax(mid_jax, x_ref)

    @jax.jit
    def run_nested(x):
      r = jax.new_ref(x)
      outer_thunky(r)
      return r[...]

    x = jnp.ones(16, dtype=jnp.float32)
    res = run_nested(x)
    self.assertArraysAllClose(np.asarray(res), np.asarray(x + const_vec))

    compiled = run_nested.lower(x).compile()
    with make_ir_context():
      mlir_mod = jax_executable_to_mlir(compiled)
    main_func = [
        op.operation
        for op in mlir_mod.body.operations
        if op.operation.name == "func.func"
    ][0]
    call_proto_ops = [
        op.operation
        for op in main_func.regions[0].blocks[0].operations
        if op.operation.name == "thunky.call_thunk_proto"
    ]
    self.assertNotEmpty(call_proto_ops)
    for cop in call_proto_ops:
      asm_text = jax._src.lib.mlir.ir.StringAttr(
          cop.attributes["asm_text"]
      ).value
      binary = jax._src.lib.mlir.ir.StringAttr(
          cop.attributes["binary"]
      ).value_bytes
      self.assertTrue(len(asm_text) > 0 or len(binary) > 0)

  def test_executable_to_mlir_constant_filtering(self):
    """Verifies jax_executable_to_mlir only attaches constant buffers to CallThunkProtoOps that reference them."""
    lut_mat = np.eye(64, dtype=np.float32) * 2.0

    @jax.jit
    def two_stage_fn(x, y):
      out1 = x @ jnp.asarray(lut_mat)
      out2 = y @ y
      return out1, out2

    compiled = two_stage_fn.lower(
        jax.ShapeDtypeStruct((64, 64), jnp.float32),
        jax.ShapeDtypeStruct((16, 16), jnp.float32),
    ).compile()
    with make_ir_context():
      mlir_mod = jax_executable_to_mlir(compiled)

    main_func = [
        op.operation
        for op in mlir_mod.body.operations
        if op.operation.name == "func.func"
    ][0]
    call_ops = [
        op.operation
        for op in main_func.regions[0].blocks[0].operations
        if op.operation.name == "thunky.call_thunk_proto"
    ]
    self.assertGreaterEqual(len(call_ops), 2)
    y_gemm_ops = [
        op
        for op in call_ops
        if any("!thunky.buffer<1024>" in str(v.type) for v in op.operands)
    ]
    self.assertNotEmpty(y_gemm_ops)
    for op in y_gemm_ops:
      operand_types = [str(v.type) for v in op.operands]
      self.assertNotIn("!thunky.buffer<16384>", operand_types)

    x_val = jnp.ones((64, 64), dtype=jnp.float32)
    y_val = jnp.ones((16, 16), dtype=jnp.float32) * 3.0
    out1, out2 = two_stage_fn(x_val, y_val)
    np.testing.assert_allclose(
        np.asarray(out1), np.ones((64, 64)) * 2.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(out2), np.ones((16, 16)) * 144.0, rtol=1e-5
    )

  # ============================================================================
  # Control Flow (`cond`, `switch`, `while_loop`)
  # ============================================================================

  def test_cond_and_switch(self):
    """Verifies thunky.cond and thunky.switch lower to XLA:GPU ConditionalThunk and execute branches."""

    def true_branch_jax(a_ref, out_ref):
      out_ref[...] = a_ref[...] * 10.0

    def false_branch_jax(a_ref, out_ref):
      out_ref[...] = a_ref[...] + 5.0

    @thunky.jit
    def cond_thunky_fn(pred_buf, x_buf, out_buf):
      def on_true():
        thunky.call_jax(true_branch_jax, x_buf, out_buf)

      def on_false():
        thunky.call_jax(false_branch_jax, x_buf, out_buf)

      thunky.cond(pred_buf, on_true, on_false)

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    out_true = jnp.zeros_like(x)
    self.assertEqual(
        _normalize_mlir(
            str(cond_thunky_fn.lower_to_mlir(jnp.array(True), x, out_true))
        ),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<1>, %arg1: !thunky.buffer<16>, %arg2: !thunky.buffer<16>, %arg3: !thunky.buffer<16>, %arg4: !thunky.buffer<16>) {
                thunky.cond %arg0 : <1> {
                  thunky.call_thunk_proto %arg1, %arg2 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                }, {
                  thunky.call_thunk_proto %arg1, %arg2 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                }
                return
              }
            }"""),
    )

    @jax.jit
    def run_cond(pred, x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      cond_thunky_fn(pred, x, out_ref)
      return out_ref[...]

    np.testing.assert_allclose(
        np.asarray(run_cond(jnp.array(True), x)),
        np.asarray(x) * 10.0,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(run_cond(jnp.array(False), x)),
        np.asarray(x) + 5.0,
        rtol=1e-5,
    )

    # Also verify multi-way branching via thunky.switch
    def branch0_jax(a_ref, out_ref):
      out_ref[...] = a_ref[...] + 1.0

    def branch1_jax(a_ref, out_ref):
      out_ref[...] = a_ref[...] + 2.0

    def branch2_jax(a_ref, out_ref):
      out_ref[...] = a_ref[...] + 3.0

    @thunky.jit
    def switch_thunky_fn(idx_buf, x_buf, out_buf):
      thunky.switch(
          idx_buf,
          [
              lambda: thunky.call_jax(branch0_jax, x_buf, out_buf),
              lambda: thunky.call_jax(branch1_jax, x_buf, out_buf),
              lambda: thunky.call_jax(branch2_jax, x_buf, out_buf),
          ],
      )

    @jax.jit
    def run_switch(idx, x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      switch_thunky_fn(idx, x, out_ref)
      return out_ref[...]

    for idx_val, offset in [(0, 1.0), (1, 2.0), (2, 3.0)]:
      np.testing.assert_allclose(
          np.asarray(run_switch(jnp.int32(idx_val), x)),
          np.asarray(x) + offset,
          rtol=1e-5,
      )

  def test_while_loop(self):
    """Verifies thunky.while_loop lowers to XLA:GPU WhileThunk and iterates."""

    def check_cond_jax(i_ref, cond_ref):
      cond_ref[...] = i_ref[...] < 3

    def step_acc_jax(acc_ref):
      acc_ref[...] += 2.0

    def step_i_jax(i_ref):
      i_ref[...] += jnp.int32(1)

    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def while_thunky_fn(x_buf, out_buf, i_buf, cond_buf):
      thunky.memzero(i_buf)
      thunky.copy(x_buf, out_buf)

      def cond_fn():
        thunky.call_jax(check_cond_jax, i_buf, cond_buf)

      def body_fn():
        thunky.call_jax(step_acc_jax, out_buf)
        thunky.call_jax(step_i_jax, i_buf)

      thunky.while_loop(cond_buf, cond_fn, body_fn)

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    out = jnp.zeros_like(x)
    self.assertEqual(
        _normalize_mlir(str(while_thunky_fn.lower_to_mlir(x, out))),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<16>, %arg1: !thunky.buffer<16>, %arg2: !thunky.buffer<4>, %arg3: !thunky.buffer<1>, %arg4: !thunky.buffer<16>) {
                thunky.memzero %arg2 : <4>
                thunky.copy %arg0, %arg1 : <16>, <16>
                thunky.while %arg3 : <1> cond {
                  thunky.call_thunk_proto %arg2, %arg3 : !thunky.buffer<4>, !thunky.buffer<1> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                } body {
                  thunky.call_thunk_proto %arg1 : !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                  thunky.call_thunk_proto %arg2 : !thunky.buffer<4> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                }
                return
              }
            }"""),
    )

    @jax.jit
    def run_while(x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      while_thunky_fn(x, out_ref)
      return out_ref[...]

    expected = np.asarray(x) + 6.0
    np.testing.assert_allclose(np.asarray(run_while(x)), expected, rtol=1e-5)

  # ============================================================================
  # Multi-Stream Concurrency (`async_start` & `async_done`)
  # ============================================================================

  def test_parallel_streams(self):
    """Verifies parallel execution across two computation streams using async_start and async_done."""

    def stream0_kernel(a_ref, out_ref):
      out_ref[...] = a_ref[...] * 3.0 + 1.0

    def stream1_kernel(b_ref, out_ref):
      out_ref[...] = b_ref[...] * 5.0 - 2.0

    @thunky.jit
    def parallel_streams_fn(x_buf, y_buf, out0_buf, out1_buf):
      def stream0_work():
        thunky.call_jax(stream0_kernel, x_buf, out0_buf)

      def stream1_work():
        thunky.call_jax(stream1_kernel, y_buf, out1_buf)

      tok0 = thunky.async_start(stream0_work, stream_id=0)
      tok1 = thunky.async_start(stream1_work, stream_id=1)
      thunky.async_done(tok0)
      thunky.async_done(tok1)

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    y = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    out0 = jnp.zeros_like(x)
    out1 = jnp.zeros_like(y)

    self.assertEqual(
        _normalize_mlir(
            str(parallel_streams_fn.lower_to_mlir(x, y, out0, out1))
        ),
        textwrap.dedent("""\
            module {
              func.func @main(%arg0: !thunky.buffer<16>, %arg1: !thunky.buffer<16>, %arg2: !thunky.buffer<16>, %arg3: !thunky.buffer<16>, %arg4: !thunky.buffer<16>, %arg5: !thunky.buffer<16>) {
                %0 = thunky.async_start stream = 0 {
                  thunky.call_thunk_proto %arg0, %arg2 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                } : !thunky.token
                %1 = thunky.async_start stream = 1 {
                  thunky.call_thunk_proto %arg1, %arg3 : !thunky.buffer<16>, !thunky.buffer<16> name = "..." {asm_text = "...", binary = "...", thunk_proto = "..."}
                } : !thunky.token
                thunky.async_done %0 : !thunky.token
                thunky.async_done %1 : !thunky.token
                return
              }
            }"""),
    )

    out0_ref = jax.new_ref(out0)
    out1_ref = jax.new_ref(out1)
    parallel_streams_fn(x, y, out0_ref, out1_ref)
    expected0 = np.asarray(x) * 3.0 + 1.0
    expected1 = np.asarray(y) * 5.0 - 2.0
    np.testing.assert_allclose(np.asarray(out0_ref[...]), expected0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out1_ref[...]), expected1, rtol=1e-5)

  # ============================================================================
  # Calling `@thunky.jit` from `@jax.jit` (State Discharge)
  # ============================================================================

  def test_calling_thunky_from_jax_jit(self):
    """Verifies calling a thunky program from @jax.jit folds thunks directly into XLA's ThunkSequence."""

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((4,), jnp.float32))
    def thunky_add_one(in_buf, out_buf, scratch):
      thunky.memzero(scratch)
      thunky.copy(in_buf, scratch)
      thunky.ptx_kernel(
          scratch,
          out_buf,
          written=[False, True],
          kernel_name="add_one_kernel",
          ptx=ADD_ONE_PTX,
          grid_dim=(1, 1, 1),
          block_dim=(4, 1, 1),
      )

    @jax.jit
    def outer_fn(x):
      y = x * 2.0
      z_ref = jax.new_ref(jnp.zeros_like(x))
      thunky_add_one(y, z_ref)
      return z_ref[...] + 5.0

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    result = outer_fn(x)
    expected = (np.asarray(x) * 2.0 + 1.0) + 5.0
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)

    # Verify XLA lowers the thunky program via thunky.inline_module
    compiled = outer_fn.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    # Verify passing a non-Ref array for a mutated parameter inside @jax.jit raises TypeError
    @jax.jit
    def bad_call(val, out_arr):
      thunky_add_one(val, out_arr)

    with self.assertRaisesRegex(TypeError, "must be passed as a Ref"):
      bad_call(x, jnp.zeros_like(x))

    # Verify state discharge with multiple slices of the same base Ref
    def accum_kernel(x_ref, y_ref):
      y_ref[...] += x_ref[...] * 2.0

    @jax.jit
    def run_in_place_sliced(x_arr, y_init):
      @thunky.jit
      def prog(x_buf, y_init_buf, y_out_buf):
        thunky.copy(y_init_buf, y_out_buf)
        thunky.call_jax(accum_kernel, x_buf.at[:4], y_out_buf.at[:4])
        thunky.call_jax(accum_kernel, x_buf.at[4:], y_out_buf.at[4:])

      y_out = jax.new_ref(jnp.zeros_like(y_init))
      prog(x_arr, y_init, y_out)
      return y_out[...]

    x_8 = jnp.arange(8, dtype=jnp.float32)
    y_8 = jnp.ones(8, dtype=jnp.float32) * 10.0
    res_sliced = run_in_place_sliced(x_8, y_8)
    np.testing.assert_allclose(
        np.asarray(res_sliced),
        np.asarray(y_8) + np.asarray(x_8) * 2.0,
        rtol=1e-5,
    )

  def test_jax_calls_thunky_calls_jax(self):
    """Verifies multi-level composition (@jax.jit -> @thunky.jit -> call_jax / nested @thunky.jit / control flow / streams)."""

    # 1. @jax.jit -> @thunky.jit -> call_jax
    def inner_jax_fn(a_ref, b_ref, out_ref):
      out_ref[...] = a_ref[...] * 3.0 + b_ref[...]

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((4,), jnp.float32))
    def middle_thunky_fn(x_buf, y_buf, out_buf, scratch):
      thunky.memzero(scratch)
      thunky.call_jax(inner_jax_fn, x_buf, y_buf, scratch)
      thunky.copy(scratch, out_buf)

    @jax.jit
    def outer_jax_fn(x, y):
      pre_x = x * 2.0
      mid_ref = jax.new_ref(jnp.zeros_like(x))
      middle_thunky_fn(pre_x, y, mid_ref)
      return mid_ref[...] - 1.0

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    y = jnp.array([0.5, 1.5, 2.5, 3.5], dtype=jnp.float32)
    result = outer_jax_fn(x, y)
    expected = ((np.asarray(x) * 2.0) * 3.0 + np.asarray(y)) - 1.0
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)

    compiled = outer_jax_fn.lower(x, y).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    # 2. @jax.jit -> nested @thunky.jit
    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((4,), jnp.float32))
    def inner_nested(src_ref, dst_ref, scratch_ref):
      def scale_shift(a_ref, s_ref):
        s_ref[...] = a_ref[...] * 2.0 + 1.0

      thunky.call_jax(scale_shift, src_ref, scratch_ref)
      thunky.copy(scratch_ref, dst_ref)

    @thunky.jit
    def outer_nested(in_ref, out_ref):
      inner_nested(in_ref.at[:4], out_ref.at[:4])
      inner_nested(in_ref.at[4:], out_ref.at[4:])

    @jax.jit
    def run_nested_folded(val):
      out_ref = jax.new_ref(jnp.zeros_like(val))
      outer_nested(val, out_ref)
      return out_ref[...]

    x_8 = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(run_nested_folded(x_8)),
        np.asarray(x_8) * 2.0 + 1.0,
        rtol=1e-5,
    )

    # 3. @jax.jit -> thunky.cond (verifying conditional_thunk in XLA executable)
    def _cond_true(a, o):
      o[...] = a[...] * 10.0

    def _cond_false(a, o):
      o[...] = a[...] + 5.0

    @thunky.jit
    def cond_thunky(pred_buf, x_buf, out_buf):
      thunky.cond(
          pred_buf,
          lambda: thunky.call_jax(_cond_true, x_buf, out_buf),
          lambda: thunky.call_jax(_cond_false, x_buf, out_buf),
      )

    @jax.jit
    def outer_cond(pred, val):
      out_ref = jax.new_ref(jnp.zeros_like(val))
      cond_thunky(pred, val, out_ref)
      return out_ref[...]

    np.testing.assert_allclose(
        np.asarray(outer_cond(jnp.array(True), x)),
        np.asarray(x) * 10.0,
        rtol=1e-5,
    )
    compiled = outer_cond.lower(jnp.array(True), x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    # 4. @jax.jit -> thunky.while_loop (verifying while_thunk in XLA executable)
    def _while_cond(i, c):
      c[...] = i[...] < 3

    def _while_add(acc):
      acc[...] = acc[...] + 2.0

    def _while_inc(i):
      i[...] = i[...] + jnp.int32(1)

    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def while_thunky(x_buf, out_buf, i_buf, cond_buf):
      thunky.memzero(i_buf)
      thunky.copy(x_buf, out_buf)
      thunky.while_loop(
          cond_buf,
          lambda: thunky.call_jax(_while_cond, i_buf, cond_buf),
          lambda: (
              thunky.call_jax(_while_add, out_buf),
              thunky.call_jax(_while_inc, i_buf),
          ),
      )

    @jax.jit
    def outer_while(val):
      out_ref = jax.new_ref(jnp.zeros_like(val))
      while_thunky(val, out_ref)
      return out_ref[...]

    np.testing.assert_allclose(
        np.asarray(outer_while(x)), np.asarray(x) + 6.0, rtol=1e-5
    )
    compiled = outer_while.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    # 5. @jax.jit -> thunky.async_start / async_done (verifying async_start_thunk)
    def _stream0_fn(a, o):
      o[...] = a[...] * 3.0 + 1.0

    def _stream1_fn(b, o):
      o[...] = b[...] * 5.0 - 2.0

    @thunky.jit
    def parallel_thunky(a_buf, b_buf, out0_buf, out1_buf):
      tok0 = thunky.async_start(
          lambda: thunky.call_jax(_stream0_fn, a_buf, out0_buf),
          stream_id=0,
      )
      tok1 = thunky.async_start(
          lambda: thunky.call_jax(_stream1_fn, b_buf, out1_buf),
          stream_id=1,
      )
      thunky.async_done(tok0)
      thunky.async_done(tok1)

    @jax.jit
    def outer_parallel(a, b):
      out0_ref = jax.new_ref(jnp.zeros_like(a))
      out1_ref = jax.new_ref(jnp.zeros_like(b))
      parallel_thunky(a, b, out0_ref, out1_ref)
      return out0_ref[...], out1_ref[...]

    res0, res1 = outer_parallel(x, y)
    np.testing.assert_allclose(
        np.asarray(res0), np.asarray(x) * 3.0 + 1.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(res1), np.asarray(y) * 5.0 - 2.0, rtol=1e-5
    )
    compiled = outer_parallel.lower(x, y).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_discharge_read_only_thunky_program(self):
    """Verifies state discharge of a read-only @thunky.jit function emits a non-empty dummy FFI output shape."""

    @thunky.jit
    def read_only_thunky(x_ref):
      thunky.ptx_kernel(
          x_ref,
          written=(False,),
          kernel_name="noop_read",
          ptx=(
              ".version 8.0\n.target sm_80\n.address_size 64\n"
              ".visible .entry noop_read(.param .u64 p0) { ret; }\n"
          ),
          grid_dim=(1, 1, 1),
          block_dim=(1, 1, 1),
          shmem_bytes=0,
      )

    def state_body(x_ref):
      read_only_thunky(x_ref)

    aval = make_buffer_aval((4,), jnp.float32)
    closed_jaxpr = jax.make_jaxpr(state_body)(aval)
    discharged_jaxpr = state_discharge.discharge_state(closed_jaxpr.jaxpr)
    ffi_eqns = [
        eqn for eqn in discharged_jaxpr.eqns if eqn.primitive.name == "ffi_call"
    ]
    self.assertNotEmpty(ffi_eqns)
    self.assertGreaterEqual(len(ffi_eqns[0].outvars), 1)

  # ============================================================================
  # XLA FFI Custom Call Thunks (`thunky.custom_call`) & Custom Hopper Kernels
  # ============================================================================

  def test_ffi_custom_call_thunk_direct_and_extracted_from_jax(self):
    """Verifies typed XLA FFI CustomCallThunk lowering (thunky.custom_call) and extraction from jax.ffi.ffi_call."""
    shape = (4, 8)
    x = jnp.arange(32, dtype=jnp.float32).reshape(shape)
    y = (jnp.arange(32, dtype=jnp.float32) + 100.0).reshape(shape)

    # 1. Direct thunky.custom_call with typed 2D f32 shapes and backend_config
    @thunky.jit
    def direct_ffi_prog(x_ref, y_ref, out_ref):
      thunky.custom_call(
          "thunky.test_concat_rows_f32",
          operands=[x_ref, y_ref],
          results=[out_ref],
          backend_config={"split_row": 2},
      )

    out1 = jnp.zeros(shape, dtype=jnp.float32)
    mlir_text_1 = str(direct_ffi_prog.lower_to_mlir(x, y, out1))
    self.assertIn("thunky.custom_call", mlir_text_1)
    self.assertIn("operand_shapes = [tensor<4x8xf32>, tensor<4x8xf32>]", mlir_text_1)
    self.assertIn("result_shapes = [tensor<4x8xf32>]", mlir_text_1)

    out1_ref = jax.new_ref(out1)
    direct_ffi_prog(x, y, out1_ref)
    expected_1 = jnp.concatenate([x[:2], y[2:]], axis=0)
    self.assertArraysAllClose(np.asarray(out1_ref[...]), np.asarray(expected_1))

    # 2. CustomCallThunk extracted by executable_to_mlir from jax.ffi.ffi_call inside thunky.call_jax
    def jax_ffi_concat(a_ref, b_ref, out_r):
      res = jax.ffi.ffi_call(
          "thunky.test_concat_rows_f32",
          jax.ShapeDtypeStruct(shape, jnp.float32),
      )(a_ref[...], b_ref[...], split_row=np.int64(3))
      out_r[...] = res

    @thunky.jit
    def extracted_ffi_prog(a_val, b_val, out_ref):
      thunky.call_jax(jax_ffi_concat, a_val, b_val, out_ref)

    out2 = jnp.zeros(shape, dtype=jnp.float32)
    mlir_text_2 = str(extracted_ffi_prog.lower_to_mlir(x, y, out2))
    self.assertIn("thunky.custom_call", mlir_text_2)
    self.assertIn('target = "thunky.test_concat_rows_f32"', mlir_text_2)
    self.assertIn("operand_shapes = [tensor<4x8xf32>, tensor<4x8xf32>]", mlir_text_2)
    self.assertIn("result_shapes = [tensor<4x8xf32>]", mlir_text_2)

    out2_ref = jax.new_ref(out2)
    extracted_ffi_prog(x, y, out2_ref)
    expected_2 = jnp.concatenate([x[:3], y[3:]], axis=0)
    self.assertArraysAllClose(np.asarray(out2_ref[...]), np.asarray(expected_2))

  def test_cusolver_potrf_ffi_custom_call(self):
    """Verifies calling JAX's registered cusolver_potrf_ffi handler via thunky.custom_call."""
    # Ensure gpu_solver FFI custom calls (including cusolver_potrf_ffi) are registered.
    _ = jnp.linalg.cholesky

    m = jnp.array(
        [
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ],
        dtype=jnp.float32,
    )

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((), jnp.int32))
    def cholesky_in_place(a_ref, info_ref):
      # Because a_ref is symmetric in memory and cuSOLVER expects column-major,
      # lower=False computes the column-major upper triangle, which is the
      # row-major lower triangle L such that A = L @ L.T.
      thunky.custom_call(
          "cusolver_potrf_ffi",
          operands=[a_ref],
          results=[a_ref, info_ref],
          backend_config={"lower": False},
      )

    @jax.jit
    def run_cholesky(a):
      a_ref = jax.new_ref(a)
      cholesky_in_place(a_ref)
      return jnp.tril(a_ref[...])

    res = run_cholesky(m)
    expected = jnp.linalg.cholesky(m)
    self.assertArraysAllClose(np.asarray(res), np.asarray(expected), rtol=1e-5)

  def test_mosaic_gpu_interleaved_with_call_jax(self):
    """Combines thunky.mosaic_gpu_kernel (TMA async copies + barriers) and thunky.call_jax, both standalone and folded inside @jax.jit."""
    dev_kind = jax.devices()[0].device_kind.lower()
    if "h100" not in dev_kind and "sm90" not in dev_kind:
      self.skipTest(f"Mosaic GPU TMA test requires Hopper GPU, got {dev_kind}")

    shape = (128, 128)
    dtype = jnp.float32

    def tma_copy_kernel(ctx, src, dst, scratch):
      smem, barrier = scratch
      ctx.async_copy(src_ref=src, dst_ref=smem, barrier=barrier)
      barrier.wait()
      ctx.async_copy(src_ref=smem, dst_ref=dst)
      ctx.await_async_copy(0)

    def postprocess(a_ref, out_ref):
      out_ref[...] = a_ref[...] * 3.0 + 1.0

    @thunky.jit(scratch_shapes=jax.ShapeDtypeStruct(shape, dtype))
    def combined_prog(in_buf, out_buf, copied_buf):
      thunky.mosaic_gpu_kernel(
          tma_copy_kernel,
          grid=(1, 1, 1),
          block=(128, 1, 1),
          in_shape=jax.ShapeDtypeStruct(shape, dtype),
          out_shape=jax.ShapeDtypeStruct(shape, dtype),
          smem_scratch_shape=(
              jax.ShapeDtypeStruct(shape, dtype),
              mgpu.TMABarrier(),
          ),
          operands=[in_buf],
          results=[copied_buf],
      )
      thunky.call_jax(postprocess, copied_buf, out_buf)

    # 1. Standalone execution
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(shape)
    out = jax.new_ref(jnp.zeros_like(x))
    combined_prog(x, out)
    expected = np.asarray(x) * 3.0 + 1.0
    np.testing.assert_allclose(np.asarray(out[...]), expected, rtol=1e-5)

    # 2. Folded inside @jax.jit
    @jax.jit
    def outer_jax_fn(val):
      pre = val + 1.0
      mid_ref = jax.new_ref(jnp.zeros(shape, dtype))
      combined_prog(pre, mid_ref)
      return mid_ref[...] - 2.0

    folded_out = outer_jax_fn(x)
    expected_folded = ((np.asarray(x) + 1.0) * 3.0 + 1.0) - 2.0
    np.testing.assert_allclose(
        np.asarray(folded_out), expected_folded, rtol=1e-5
    )

  # ============================================================================
  # Multi-GPU Collectives & Compute/Communication Overlap
  # ============================================================================

  def test_all_reduce(self):
    """Verifies thunky.all_reduce executes AllReduceThunk across 2 GPUs."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def ar_thunky(src_buf, dst_buf):
      thunky.all_reduce(
          src_buf,
          dst_buf,
          reduction="sum",
          replica_groups=((0, 1),),
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_ar(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      ar_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_ar(x)
    expected = np.array(
        [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_ar.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_gather(self):
    """Verifies thunky.all_gather executes AllGatherThunk across 2 GPUs."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def ag_thunky(src_buf, dst_buf):
      thunky.all_gather(
          src_buf,
          dst_buf,
          replica_groups=((0, 1),),
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_ag(x_local):
      out_ref = jax.new_ref(jnp.zeros((1, 8), dtype=x_local.dtype))
      ag_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_ag(x)
    expected_row = np.array(
        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32
    )
    expected = np.stack([expected_row, expected_row], axis=0)
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_ag.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_reduce_scatter(self):
    """Verifies thunky.reduce_scatter executes ReduceScatterThunk across 2 GPUs."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def rs_thunky(src_buf, dst_buf):
      thunky.reduce_scatter(
          src_buf,
          dst_buf,
          reduction="sum",
          replica_groups=((0, 1),),
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_rs(x_local):
      out_ref = jax.new_ref(jnp.zeros((1, 4), dtype=x_local.dtype))
      rs_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ],
        dtype=jnp.float32,
    )
    res = run_rs(x)
    expected = np.array(
        [[11.0, 22.0, 33.0, 44.0], [55.0, 66.0, 77.0, 88.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_rs.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_to_all(self):
    """Verifies thunky.all_to_all executes AllToAllThunk across 2 GPUs."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def a2a_thunky(src_buf, dst_buf):
      thunky.all_to_all(
          src_buf,
          dst_buf,
          replica_groups=((0, 1),),
          has_split_dimension=True,
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_a2a(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      a2a_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=jnp.float32,
    )
    res = run_a2a(x)
    expected = np.array(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_a2a.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_to_all_split_dimension_and_buffer_sequences(self):
    """Verifies all_to_all validates has_split_dimension=False on single buffers and supports sequences of buffers per rank."""

    @thunky.jit
    def bad_all_to_all(src_ref, dst_ref):
      thunky.all_to_all(
          src_ref,
          dst_ref,
          replica_groups=((0, 1),),
          has_split_dimension=False,
      )

    aval = make_buffer_aval((4,), jnp.float32)
    with self.assertRaises(ValueError):
      bad_all_to_all.trace_jaxpr(aval, aval)

    @thunky.jit
    def tuple_all_to_all(src0_ref, src1_ref, dst0_ref, dst1_ref):
      thunky.all_to_all(
          (src0_ref, src1_ref),
          (dst0_ref, dst1_ref),
          replica_groups=((0, 1),),
          has_split_dimension=False,
      )

    tuple_all_to_all.trace_jaxpr(aval, aval, aval, aval)
    tuple_all_to_all.lower_to_mlir(aval, aval, aval, aval)
    if len(jax.devices()) >= 2:
      mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

      @jax.jit
      @shard_map(
          mesh=mesh,
          in_specs=(P("x"), P("x")),
          out_specs=(P("x"), P("x")),
          check_vma=False,
      )
      def run_tuple_a2a(s0, s1):
        r_s0 = jax.new_ref(s0)
        r_s1 = jax.new_ref(s1)
        r_d0 = jax.new_ref(jnp.zeros_like(s0))
        r_d1 = jax.new_ref(jnp.zeros_like(s1))
        tuple_all_to_all(r_s0, r_s1, r_d0, r_d1)
        return r_d0[...], r_d1[...]

      x0 = jnp.array([[1.0, 2.0], [10.0, 20.0]], dtype=jnp.float32)
      x1 = jnp.array([[3.0, 4.0], [30.0, 40.0]], dtype=jnp.float32)
      y0, y1 = run_tuple_a2a(x0, x1)
      expected_y0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
      expected_y1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
      self.assertArraysAllClose(np.asarray(y0), expected_y0)
      self.assertArraysAllClose(np.asarray(y1), expected_y1)

  def test_collective_permute(self):
    """Verifies thunky.collective_permute executes CollectivePermuteThunk across 2 GPUs."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def cp_thunky(src_buf, dst_buf):
      thunky.collective_permute(
          src_buf,
          dst_buf,
          source_target_pairs=((0, 1), (1, 0)),
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_cp(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      cp_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_cp(x)
    expected = np.array(
        [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_cp.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_collective_permute_with_non_participating_ranks(self):
    """Verifies thunky.collective_permute supports replica_groups / axis_name including non-participating ranks."""
    sig = inspect.signature(thunky.collective_permute)
    self.assertTrue(
        "replica_groups" in sig.parameters or "axis_name" in sig.parameters
    )

    @thunky.jit
    def permute_subset(src_ref, dst_ref):
      thunky.collective_permute(
          src_ref,
          dst_ref,
          source_target_pairs=((0, 1),),
          replica_groups=((0, 1, 2, 3),),
      )

    aval = make_buffer_aval((4,), jnp.float32)
    module = permute_subset.lower_to_mlir(aval, aval)
    main_func = [
        op.operation
        for op in module.body.operations
        if op.operation.name == "func.func"
    ][0]
    cp_ops = [
        op.operation
        for op in main_func.regions[0].blocks[0].operations
        if op.operation.name == "thunky.collective_permute"
    ]
    self.assertLen(cp_ops, 1)
    self.assertIn("replica_groups", cp_ops[0].attributes)

    if len(jax.devices()) >= 2:
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("x",))

      @thunky.jit
      def permute_one_way(src_ref, dst_ref):
        thunky.collective_permute(
            src_ref,
            dst_ref,
            source_target_pairs=((0, 1),),
            replica_groups=((0, 1),),
        )

      @jax.jit
      @shard_map(
          mesh=mesh,
          in_specs=(P("x"), P("x")),
          out_specs=P("x"),
          check_vma=False,
      )
      def run_one_way(src, dst):
        s_ref = jax.new_ref(src)
        d_ref = jax.new_ref(dst)
        permute_one_way(s_ref, d_ref)
        return d_ref[...]

      src_arr = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 10.0
      dst_arr = jnp.zeros((2, 4), dtype=jnp.float32)
      out = run_one_way(src_arr, dst_arr)
      self.assertArraysAllClose(np.asarray(out[1]), np.asarray(src_arr[0]))

  def test_call_jax_with_collective(self):
    """Verifies JaxExecutableToMlir and call_jax round-trip a JAX function containing collectives."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    def jax_collective_fn(a_ref, out_ref):
      out_ref[...] = jax.lax.psum(a_ref[...], "x") * 2.0

    @jax.jit
    def jitted_jax_collective_fn(a_ref, out_ref):
      out_ref[...] = jax.lax.psum(a_ref[...], "x") * 2.0

    @thunky.jit
    def splice_collective_thunky(x_buf, out_buf):
      thunky.call_jax(jax_collective_fn, x_buf, out_buf)

    @thunky.jit
    def splice_jitted_collective_thunky(x_buf, out_buf):
      thunky.call_jax(jitted_jax_collective_fn, x_buf, out_buf)

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_spliced(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      splice_collective_thunky(x_local, out_ref)
      return out_ref[...]

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_spliced_jitted(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      splice_jitted_collective_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    expected = np.array(
        [[22.0, 44.0, 66.0, 88.0], [22.0, 44.0, 66.0, 88.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(run_spliced(x)), expected, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(run_spliced_jitted(x)), expected, rtol=1e-5
    )

  def test_ref_assignment_with_collectives_in_manual_mesh(self):
    """Verifies direct Ref assignment with collectives (swap_p) preserves the active manual mesh axis environment."""
    abstract_mesh = jax.sharding.AbstractMesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Manual,)
    )

    @thunky.jit
    def swap_with_collective(x_ref, out_ref):
      out_ref[...] = jax.lax.psum(x_ref[...], "x")

    aval = make_buffer_aval((4,), jnp.float32)
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      swap_with_collective.lower_to_mlir(aval, aval)

  def test_call_jax_with_large_abstract_mesh(self):
    """Verifies call_jax lowers using AbstractMesh directly even when mesh size exceeds local device count."""
    large_abstract_mesh = jax.sharding.AbstractMesh(
        (16,), ("x",), axis_types=(jax.sharding.AxisType.Manual,)
    )

    def _add_one(r):
      r[...] = r[...] + 1.0

    @thunky.jit
    def prog(x_ref):
      thunky.call_jax(_add_one, x_ref)

    aval = make_buffer_aval((4,), jnp.float32)
    with jax.sharding.use_abstract_mesh(large_abstract_mesh):
      prog.lower_to_mlir(aval)

  def test_executable_to_mlir_multi_operand_collectives(self):
    """Verifies jax_executable_to_mlir extracts all buffers from combined multi-operand collective thunks."""
    abstract_mesh = jax.sharding.AbstractMesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Manual,)
    )

    @jax.jit
    @shard_map(
        mesh=abstract_mesh,
        in_specs=(P(), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def two_pperms(a, b):
      return (
          jax.lax.ppermute(a, "x", perm=[(0, 1), (1, 0)]),
          jax.lax.ppermute(b, "x", perm=[(0, 1), (1, 0)]),
      )

    compiled = two_pperms.lower(
        jax.ShapeDtypeStruct((16,), jnp.float32),
        jax.ShapeDtypeStruct((32,), jnp.float32),
    ).compile()
    with make_ir_context():
      mlir_mod = jax_executable_to_mlir(compiled)

    main_func = [
        op.operation
        for op in mlir_mod.body.operations
        if op.operation.name == "func.func"
    ][0]
    cp_ops = [
        op.operation
        for op in main_func.regions[0].blocks[0].operations
        if op.operation.name == "thunky.collective_permute"
    ]
    self.assertLen(cp_ops, 2)

    if len(jax.devices()) >= 2:
      mesh = jax.sharding.Mesh(jax.devices()[:2], ("x",))

      def _jax_two_pperms(a_ref, b_ref, out_a_ref, out_b_ref):
        out_a_ref[...] = jax.lax.ppermute(
            a_ref[...], "x", perm=[(0, 1), (1, 0)]
        )
        out_b_ref[...] = jax.lax.ppermute(
            b_ref[...], "x", perm=[(0, 1), (1, 0)]
        )

      @thunky.jit
      def thunky_two_pperms(a_buf, b_buf, out_a_buf, out_b_buf):
        thunky.call_jax(_jax_two_pperms, a_buf, b_buf, out_a_buf, out_b_buf)

      @jax.jit
      @shard_map(
          mesh=mesh,
          in_specs=(P("x"), P("x")),
          out_specs=(P("x"), P("x")),
          check_vma=False,
      )
      def run_on_mesh(a, b):
        out_a = jax.new_ref(jnp.zeros((16,), dtype=jnp.float32))
        out_b = jax.new_ref(jnp.zeros((32,), dtype=jnp.float32))
        thunky_two_pperms(jax.new_ref(a), jax.new_ref(b), out_a, out_b)
        return out_a[...], out_b[...]

      a_in = jnp.arange(32, dtype=jnp.float32)
      b_in = jnp.arange(64, dtype=jnp.float32) * 10.0
      res_a, res_b = run_on_mesh(a_in, b_in)
      np.testing.assert_allclose(
          np.asarray(res_a[:16]), np.asarray(a_in[16:]), rtol=1e-5
      )
      np.testing.assert_allclose(
          np.asarray(res_b[:32]), np.asarray(b_in[32:]), rtol=1e-5
      )

  def test_non_manual_mesh_raises_error(self):
    """Verifies calling a thunky function in a non-manual mesh environment raises ValueError."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def kernel(a_ref, out_ref):
      thunky.copy(a_ref, out_ref)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "manual mode"):
      with jax.set_mesh(mesh):
        out_ref = jax.new_ref(jnp.zeros_like(x))
        kernel(x, out_ref)

  def test_async_collective_compute_overlap(self):
    """Verifies overlapping a collective on a communication stream with compute on a computation stream."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    def compute_kernel(y_ref, out_ref):
      out_ref[...] = y_ref[...] * 3.0 + 1.0

    @thunky.jit
    def overlap_thunky(x_buf, y_buf, ar_out_buf, comp_out_buf):
      def comm_work():
        thunky.all_reduce(
            x_buf,
            ar_out_buf,
            reduction="sum",
            replica_groups=((0, 1),),
        )

      # Launch AllReduce asynchronously on the communication stream
      comm_tok = thunky.async_start(
          comm_work, stream_id=0, stream_kind="communication"
      )
      # Run compute kernel concurrently on the main execution stream
      thunky.call_jax(compute_kernel, y_buf, comp_out_buf)
      # Synchronize communication stream back to the main stream
      thunky.async_done(comm_tok)

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_overlap(x_local, y_local):
      ar_out_ref = jax.new_ref(jnp.zeros_like(x_local))
      comp_out_ref = jax.new_ref(jnp.zeros_like(y_local))
      overlap_thunky(x_local, y_local, ar_out_ref, comp_out_ref)
      return ar_out_ref[...], comp_out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    y = jnp.array(
        [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]], dtype=jnp.float32
    )
    ar_res, comp_res = run_overlap(x, y)

    expected_ar = np.array(
        [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]], dtype=np.float32
    )
    expected_comp = np.asarray(y) * 3.0 + 1.0
    np.testing.assert_allclose(np.asarray(ar_res), expected_ar, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(comp_res), expected_comp, rtol=1e-5)

    compiled = run_overlap.lower(x, y).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())
    mlir_text = jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.all_reduce", mlir_text)
    self.assertIn("is_communication = true", mlir_text)

  def test_ring_all_gather_collective_matmul(self):
    """Verifies an N-GPU Ring AllGather(A) @ B collective GEMM overlapping CollectivePermute with matmul into sliced output buffers."""
    num_devices = len(jax.devices())
    if num_devices < 2:
      self.skipTest("Requires at least 2 GPUs")

    mesh = Mesh(np.array(jax.devices()), ("x",))
    m_block = int(os.environ.get("THUNKY_M_BLOCK", "8"))
    k_dim = int(os.environ.get("THUNKY_K_DIM", "32"))
    n_block = int(os.environ.get("THUNKY_N_BLOCK", "8"))
    avoid_d2h = os.environ.get("THUNKY_AVOID_D2H", "0") == "1"
    multistream = os.environ.get("THUNKY_MULTISTREAM", "0") == "1"
    dtype_str = os.environ.get("THUNKY_DTYPE", "bf16")
    dtype = {"f32": jnp.float32, "bf16": jnp.bfloat16, "f16": jnp.float16}[
        dtype_str
    ]

    def matmul_fn(lhs_ref, rhs_ref, out_ref):
      out_ref[...] = jnp.matmul(lhs_ref[...], rhs_ref[...])

    if multistream:

      @thunky.jit(
          scratch_shapes=(
              [
                  jax.ShapeDtypeStruct((m_block, k_dim), dtype)
                  for _ in range(num_devices - 1)
              ],
              jax.ShapeDtypeStruct((num_devices * m_block, n_block), dtype),
          )
      )
      def ring_all_gather_matmul_thunky(
          a_local_buf,
          b_local_buf,
          dev_id_buf,
          c_out_buf,
          a_scratch_bufs,
          c_ring_buf,
      ):
        bufs = [a_local_buf] + list(a_scratch_bufs)
        c_ring_slices = [
            c_ring_buf.at[s * m_block : (s + 1) * m_block]
            for s in range(num_devices)
        ]
        ring_pairs = tuple(
            (i, (i + 1) % num_devices) for i in range(num_devices)
        )

        tok_a = {}
        tok_c = {}

        def start_comm(step_idx):
          return thunky.async_start(
              lambda src=bufs[step_idx - 1], dst=bufs[step_idx]: (
                  thunky.collective_permute(
                      src, dst, source_target_pairs=ring_pairs
                  )
              ),
              stream_id=0,
              stream_kind="communication",
          )

        def start_gemm(step_idx):
          return thunky.async_start(
              lambda a_buf=bufs[step_idx], out_slice=c_ring_slices[step_idx]: (
                  thunky.call_jax(matmul_fn, a_buf, b_local_buf, out_slice)
              ),
              stream_id=(step_idx % 2),
              stream_kind="computation",
          )

        # Queue all N-1 collective permutes back-to-back on communication stream
        for s in range(1, num_devices):
          tok_a[s] = start_comm(s)

        # Launch Gemm 0 on computeA immediately
        tok_c[0] = start_gemm(0)

        # Each subsequent Gemm s waits only on tok_a[s] (arrival of bufs[s])
        for s in range(1, num_devices):
          thunky.async_done(tok_a[s])
          tok_c[s] = start_gemm(s)

        # Wait for all compute streams before final coalesced reorder
        for s in range(num_devices):
          thunky.async_done(tok_c[s])

        def unroll_fn(c_ring_ref, dev_id_ref, c_out_ref):
          d = dev_id_ref[...]
          indices = (d - jnp.arange(num_devices, dtype=jnp.int32)) % num_devices
          c_out_ref[...] = (
              c_ring_ref[...]
              .reshape(num_devices, m_block, n_block)[indices]
              .reshape(m_block * num_devices, n_block)
          )

        thunky.call_jax(unroll_fn, c_ring_buf, dev_id_buf, c_out_buf)

    else:

      @thunky.jit(
          scratch_shapes=(
              [jax.ShapeDtypeStruct((m_block, k_dim), dtype) for _ in range(2)],
          )
      )
      def ring_all_gather_matmul_thunky(
          a_local_buf, b_local_buf, dev_id_buf, c_out_buf, scratch_bufs
      ):
        # Zero-copy views into the N row blocks of c_out_buf (total shape (m_block * N, n_block))
        c_slices = [
            c_out_buf.at[r * m_block : (r + 1) * m_block]
            for r in range(num_devices)
        ]
        ring_pairs = tuple(
            (i, (i + 1) % num_devices) for i in range(num_devices)
        )

        for step in range(num_devices):
          curr_a = a_local_buf if step == 0 else scratch_bufs[(step - 1) % 2]

          # Overlap CollectivePermute for the next ring step with the current step's GEMM
          if step < num_devices - 1:
            next_a = scratch_bufs[step % 2]
            comm_tok = thunky.async_start(
                lambda src=curr_a, dst=next_a: thunky.collective_permute(
                    src, dst, source_target_pairs=ring_pairs
                ),
                stream_id=0,
                stream_kind="communication",
            )

          if avoid_d2h:
            # Compute GEMM and write into row slice `(dev_id - step) % num_devices` on device
            # without host-device synchronization (avoiding ConditionalThunk MemcpyD2H).
            def make_step_matmul_fn(s):
              def _step_fn(lhs_ref, rhs_ref, dev_id_ref, c_out_ref):
                block = jnp.matmul(lhs_ref[...], rhs_ref[...])
                row_idx = (dev_id_ref[...] - s) % num_devices
                c_out_ref[...] = jax.lax.dynamic_update_slice(
                    c_out_ref[...], block, (row_idx * m_block, 0)
                )

              return _step_fn

            thunky.call_jax(
                make_step_matmul_fn(step),
                curr_a,
                b_local_buf,
                dev_id_buf,
                c_out_buf,
            )
          else:
            # At step `step`, device `d` holds row block `(d - step) % num_devices` of A.
            # Switch on dev_id_buf (int32) to write GEMM output directly into that row slice of C.
            thunky.switch(
                dev_id_buf,
                [
                    lambda c_slice=c_slices[
                        (d - step) % num_devices
                    ], a_buf=curr_a: (
                        thunky.call_jax(matmul_fn, a_buf, b_local_buf, c_slice)
                    )
                    for d in range(num_devices)
                ],
            )

          if step < num_devices - 1:
            thunky.async_done(comm_tok)

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x", None), P(None, "x")),
        out_specs=P(None, "x"),
        check_vma=False,
    )
    def run_collective_matmul(a_local, b_local):
      dev_id = jnp.int32(jax.lax.axis_index("x"))
      c_out_ref = jax.new_ref(
          jnp.zeros((m_block * num_devices, n_block), dtype=dtype)
      )
      ring_all_gather_matmul_thunky(a_local, b_local, dev_id, c_out_ref)
      return c_out_ref[...]

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(
        (m_block * num_devices, k_dim), dtype=np.float32
    ).astype(dtype)
    b_np = rng.standard_normal(
        (k_dim, n_block * num_devices), dtype=np.float32
    ).astype(dtype)

    a = jax.device_put(a_np, jax.sharding.NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b_np, jax.sharding.NamedSharding(mesh, P(None, "x")))

    res = run_collective_matmul(a, b)
    jax.block_until_ready(res)

    trace_dir = os.environ.get("JAX_TRACE_DIR")
    if trace_dir:
      with jax.profiler.trace(trace_dir):
        for _ in range(5):
          res = run_collective_matmul(a, b)
          jax.block_until_ready(res)

    expected = np.asarray(a_np, dtype=np.float32) @ np.asarray(
        b_np, dtype=np.float32
    )
    if dtype == jnp.float32:
      tol = 1e-4 * np.sqrt(k_dim / 32.0)
    else:
      tol = 2e-2 * np.sqrt(k_dim / 32.0)
    np.testing.assert_allclose(
        np.asarray(res, dtype=np.float32), expected, rtol=tol, atol=tol
    )

    compiled = run_collective_matmul.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())
    mlir_text = jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.collective_permute", mlir_text)
    self.assertIn("is_communication = true", mlir_text)
    if not avoid_d2h and not multistream:
      self.assertIn("thunky.cond", mlir_text)

  # ============================================================================
  # Nested Control Flow & Inner jax.jit Control Flow
  # ============================================================================

  def test_nested_cond_in_cond(self):
    """Tests nested thunky.cond inside thunky.cond across all branch combinations."""

    def add_val(val):
      def _fn(buf):
        buf[...] = buf[...] + val

      return _fn

    @thunky.jit
    def nested_cond_prog(p0_buf, p1_buf, x_buf, out_buf):
      thunky.copy(x_buf, out_buf)
      thunky.cond(
          p0_buf,
          lambda: thunky.cond(
              p1_buf,
              lambda: thunky.call_jax(add_val(10.0), out_buf),
              lambda: thunky.call_jax(add_val(20.0), out_buf),
          ),
          lambda: thunky.cond(
              p1_buf,
              lambda: thunky.call_jax(add_val(30.0), out_buf),
              lambda: thunky.call_jax(add_val(40.0), out_buf),
          ),
      )

    @jax.jit
    def run_nested_cond(p0, p1, x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      nested_cond_prog(p0, p1, x, out_ref)
      return out_ref[...]

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    cases = [
        (True, True, 10.0),
        (True, False, 20.0),
        (False, True, 30.0),
        (False, False, 40.0),
    ]
    for p0, p1, expected_offset in cases:
      res = run_nested_cond(jnp.array(p0), jnp.array(p1), x)
      np.testing.assert_allclose(
          np.asarray(res), np.asarray(x) + expected_offset, rtol=1e-5
      )

    mlir_mod = nested_cond_prog.lower_to_mlir(
        jnp.array(True), jnp.array(True), x, x
    )
    mlir_str = str(mlir_mod)
    self.assertEqual(mlir_str.count("thunky.cond"), 3)

    compiled = run_nested_cond.lower(
        jnp.array(True), jnp.array(True), x
    ).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_nested_while_in_while(self):
    """Tests a nested thunky.while_loop inside another thunky.while_loop."""

    def check_lt(limit):
      def _fn(idx_ref, cond_ref):
        cond_ref[...] = idx_ref[...] < jnp.int32(limit)

      return _fn

    def inc_int(idx_ref):
      idx_ref[...] = idx_ref[...] + jnp.int32(1)

    def add_product(i_ref, j_ref, out_ref):
      factor = (i_ref[...] + 1).astype(jnp.float32) * (j_ref[...] + 1).astype(
          jnp.float32
      )
      out_ref[...] = out_ref[...] + factor

    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def nested_while_prog(x_buf, out_buf, i_buf, j_buf, cond_i_buf, cond_j_buf):
      thunky.copy(x_buf, out_buf)
      thunky.memzero(i_buf)

      def outer_body():
        thunky.memzero(j_buf)
        thunky.while_loop(
            cond_j_buf,
            lambda: thunky.call_jax(check_lt(3), j_buf, cond_j_buf),
            lambda: (
                thunky.call_jax(add_product, i_buf, j_buf, out_buf),
                thunky.call_jax(inc_int, j_buf),
            ),
        )
        thunky.call_jax(inc_int, i_buf)

      thunky.while_loop(
          cond_i_buf,
          lambda: thunky.call_jax(check_lt(2), i_buf, cond_i_buf),
          outer_body,
      )

    @jax.jit
    def run_nested_while(x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      nested_while_prog(x, out_ref)
      return out_ref[...]

    x = jnp.array([5.0, 10.0, 15.0, 20.0], dtype=jnp.float32)
    res = run_nested_while(x)
    # Sum over i in {0, 1}, j in {0, 1, 2} of (i+1)*(j+1) = (1 + 2) * (1 + 2 + 3) = 3 * 6 = 18.0
    np.testing.assert_allclose(np.asarray(res), np.asarray(x) + 18.0, rtol=1e-5)

    mlir_str = str(nested_while_prog.lower_to_mlir(x, x))
    self.assertEqual(mlir_str.count("thunky.while"), 2)

    compiled = run_nested_while.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_nested_cond_and_while_mixtures(self):
    """Tests cond inside while_loop and while_loop inside cond."""

    def check_lt(limit):
      def _fn(idx_ref, cond_ref):
        cond_ref[...] = idx_ref[...] < jnp.int32(limit)

      return _fn

    def is_even(idx_ref, pred_ref):
      pred_ref[...] = (idx_ref[...] % jnp.int32(2)) == jnp.int32(0)

    def inc_int(idx_ref):
      idx_ref[...] = idx_ref[...] + jnp.int32(1)

    def add_const(c):
      def _fn(buf):
        buf[...] = buf[...] + c

      return _fn

    def mul_const(c):
      def _fn(buf):
        buf[...] = buf[...] * c

      return _fn

    # Mixture 1: cond inside while_loop (4 steps: step 0->+3, step 1->*2, step 2->+3, step 3->*2)
    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def cond_in_while_prog(
        x_buf, out_buf, step_buf, loop_cond_buf, branch_pred_buf
    ):
      thunky.copy(x_buf, out_buf)
      thunky.memzero(step_buf)

      def loop_body():
        thunky.call_jax(is_even, step_buf, branch_pred_buf)
        thunky.cond(
            branch_pred_buf,
            lambda: thunky.call_jax(add_const(3.0), out_buf),
            lambda: thunky.call_jax(mul_const(2.0), out_buf),
        )
        thunky.call_jax(inc_int, step_buf)

      thunky.while_loop(
          loop_cond_buf,
          lambda: thunky.call_jax(check_lt(4), step_buf, loop_cond_buf),
          loop_body,
      )

    # Mixture 2: while_loop inside cond
    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def while_in_cond_prog(mode_buf, x_buf, out_buf, idx_buf, loop_cond_buf):
      thunky.copy(x_buf, out_buf)
      thunky.memzero(idx_buf)
      thunky.cond(
          mode_buf,
          lambda: thunky.while_loop(
              loop_cond_buf,
              lambda: thunky.call_jax(check_lt(3), idx_buf, loop_cond_buf),
              lambda: (
                  thunky.call_jax(add_const(5.0), out_buf),
                  thunky.call_jax(inc_int, idx_buf),
              ),
          ),
          lambda: thunky.while_loop(
              loop_cond_buf,
              lambda: thunky.call_jax(check_lt(2), idx_buf, loop_cond_buf),
              lambda: (
                  thunky.call_jax(mul_const(3.0), out_buf),
                  thunky.call_jax(inc_int, idx_buf),
              ),
          ),
      )

    @jax.jit
    def run_mixtures(mode, x):
      mid_ref = jax.new_ref(jnp.zeros_like(x))
      out_ref = jax.new_ref(jnp.zeros_like(x))
      cond_in_while_prog(x, mid_ref)
      while_in_cond_prog(mode, mid_ref, out_ref)
      return out_ref[...]

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    # After cond_in_while(x):
    # step 0 (even): x + 3
    # step 1 (odd): (x + 3) * 2
    # step 2 (even): (x + 3) * 2 + 3
    # step 3 (odd): ((x + 3) * 2 + 3) * 2 = 4*x + 18
    mid_expected = 4.0 * np.asarray(x) + 18.0

    # When mode=True: adds 5.0 three times -> mid_expected + 15.0
    res_true = run_mixtures(jnp.array(True), x)
    np.testing.assert_allclose(
        np.asarray(res_true), mid_expected + 15.0, rtol=1e-5
    )

    # When mode=False: multiplies by 3.0 twice -> mid_expected * 9.0
    res_false = run_mixtures(jnp.array(False), x)
    np.testing.assert_allclose(
        np.asarray(res_false), mid_expected * 9.0, rtol=1e-5
    )

    compiled = run_mixtures.lower(jnp.array(True), x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_collatz_while_and_cond(self):
    """Computes Collatz stopping time using nested thunky.while_loop and thunky.cond."""

    def check_gt_one(n_ref, cond_ref):
      cond_ref[...] = n_ref[...] > jnp.int32(1)

    def check_is_even(n_ref, even_ref):
      even_ref[...] = (n_ref[...] % jnp.int32(2)) == jnp.int32(0)

    def collatz_even(n_ref):
      n_ref[...] = n_ref[...] // jnp.int32(2)

    def collatz_odd(n_ref):
      n_ref[...] = jnp.int32(3) * n_ref[...] + jnp.int32(1)

    def increment_steps(steps_ref):
      steps_ref[...] = steps_ref[...] + jnp.int32(1)

    @thunky.jit(
        scratch_shapes=(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
    )
    def collatz_length(
        n_in_buf, steps_out_buf, n_buf, loop_cond_buf, is_even_buf
    ):
      thunky.copy(n_in_buf, n_buf)
      thunky.memzero(steps_out_buf)

      def loop_body():
        thunky.call_jax(check_is_even, n_buf, is_even_buf)
        thunky.cond(
            is_even_buf,
            lambda: thunky.call_jax(collatz_even, n_buf),
            lambda: thunky.call_jax(collatz_odd, n_buf),
        )
        thunky.call_jax(increment_steps, steps_out_buf)

      thunky.while_loop(
          loop_cond_buf,
          lambda: thunky.call_jax(check_gt_one, n_buf, loop_cond_buf),
          loop_body,
      )

    @jax.jit
    def run_collatz(n):
      steps_ref = jax.new_ref(jnp.int32(0))
      collatz_length(n, steps_ref)
      return steps_ref[...]

    for n_val, expected_steps in [(1, 0), (6, 8), (19, 20), (27, 111)]:
      self.assertEqual(int(run_collatz(jnp.int32(n_val))), expected_steps)

  def test_whiles_and_conds_inside_inner_jax_jit(self):
    """Tests splicing inner jax.jit functions containing lax.while_loop and lax.cond into thunky."""

    @jax.jit
    def inner_jax_control_flow(pred_ref, x_ref, out_ref):
      p = pred_ref[...]
      val = x_ref[...]

      def true_branch(v):
        # lax.while_loop inside lax.cond
        def cond_fun(state):
          i, _ = state
          return i < 4

        def body_fun(state):
          i, acc = state
          # lax.cond inside lax.while_loop
          next_acc = jax.lax.cond(
              (i % 2) == 0,
              lambda a: a + 10.0,
              lambda a: a * 2.0,
              acc,
          )
          return i + 1, next_acc

        _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (jnp.int32(0), v))
        return final_acc

      def false_branch(v):
        # lax.fori_loop (lowered to WhileThunk) inside lax.cond false branch
        return jax.lax.fori_loop(0, 3, lambda i, acc: acc - 5.0, v)

      out_ref[...] = jax.lax.cond(p, true_branch, false_branch, val)

    @thunky.jit
    def thunky_calling_jax_control_flow(pred_buf, x_buf, out_buf):
      thunky.memzero(out_buf)
      thunky.call_jax(inner_jax_control_flow, pred_buf, x_buf, out_buf)

    @jax.jit
    def outer_runner(pred, x):
      out_ref = jax.new_ref(jnp.zeros_like(x))
      thunky_calling_jax_control_flow(pred, x, out_ref)
      return out_ref[...]

    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    # Verify that jax_executable_to_mlir / lower_to_mlir translates the inner jax.jit's
    # ConditionalThunk and WhileThunk into thunky.cond and thunky.while MLIR ops!
    mlir_mod = thunky_calling_jax_control_flow.lower_to_mlir(
        jnp.array(True), x, x
    )
    mlir_str = str(mlir_mod)
    self.assertIn("thunky.cond", mlir_str)
    self.assertIn("thunky.while", mlir_str)

    # True branch:
    # i=0 (even): v + 10
    # i=1 (odd): (v + 10) * 2
    # i=2 (even): (v + 10) * 2 + 10
    # i=3 (odd): ((v + 10) * 2 + 10) * 2 = 4*v + 60
    res_true = outer_runner(jnp.array(True), x)
    np.testing.assert_allclose(
        np.asarray(res_true), 4.0 * np.asarray(x) + 60.0, rtol=1e-5
    )

    # False branch: subtracts 5.0 three times -> v - 15.0
    res_false = outer_runner(jnp.array(False), x)
    np.testing.assert_allclose(
        np.asarray(res_false), np.asarray(x) - 15.0, rtol=1e-5
    )

    compiled = outer_runner.lower(jnp.array(True), x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  # ============================================================================
  # Collective Group Thunk (kGroup)
  # ============================================================================

  def test_collective_group_rejects_non_collectives(self):
    """Verifies that thunky.collective_group rejects non-collective operations."""

    @thunky.jit
    def invalid_group_prog(a_buf, out_buf):
      thunky.collective_group(lambda: thunky.copy(a_buf, out_buf))

    a = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    with self.assertRaisesRegex(
        ValueError, "can only contain collective operations"
    ):
      invalid_group_prog.lower_to_mlir(a, a)

  def test_collective_group_multi_gpu(self):
    """Tests thunky.collective_group fusing multi-GPU collectives in a single NCCL group."""
    if jax.device_count() < 2:
      self.skipTest("Requires at least 2 GPU devices")

    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices), ("x",))

    @thunky.jit
    def grouped_collectives_prog(a_buf, b_buf, out_a_buf, out_b_buf):
      thunky.collective_group(
          lambda: (
              thunky.all_reduce(
                  a_buf, out_a_buf, reduction="sum", replica_groups=((0, 1),)
              ),
              thunky.collective_permute(
                  b_buf, out_b_buf, source_target_pairs=((0, 1), (1, 0))
              ),
          )
      )

    a_slice = jnp.array([1.0, 2.0], dtype=jnp.float32)
    b_slice = jnp.array([10.0, 20.0], dtype=jnp.float32)
    mlir_mod = grouped_collectives_prog.lower_to_mlir(
        a_slice, b_slice, a_slice, b_slice
    )
    self.assertIn("thunky.collective_group", str(mlir_mod))

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_grouped_collectives(a_local, b_local):
      out_a = jax.new_ref(jnp.zeros_like(a_local))
      out_b = jax.new_ref(jnp.zeros_like(b_local))
      grouped_collectives_prog(a_local, b_local, out_a, out_b)
      return out_a[...], out_b[...]

    # Shard across 2 GPUs: device 0 has [1, 2], device 1 has [3, 4]
    a = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    # device 0 has [10, 20], device 1 has [30, 40]
    b = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)

    res_a, res_b = run_grouped_collectives(a, b)
    # all_reduce sum: both devices get [1+3, 2+4] = [4, 6]
    expected_a = np.array([4.0, 6.0, 4.0, 6.0], dtype=np.float32)
    # collective_permute swap: device 0 gets [30, 40], device 1 gets [10, 20]
    expected_b = np.array([30.0, 40.0, 10.0, 20.0], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(res_a), expected_a, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(res_b), expected_b, rtol=1e-5)

    compiled = run_grouped_collectives.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    mlir_text = jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.collective_group", mlir_text)

  def test_call_jax_arbitrary_programs(self):
    """Verifies that arbitrary jax.jit programs (cuDNN, cuFFT, cuSOLVER, TopK/Sort, constants, sliced views) work inside thunky.call_jax."""
    # 1. cuDNN convolution + large constant table + FFT + Triangular solve + Sort
    lut_np = np.linspace(-1.0, 1.0, 256, dtype=np.float32)

    def complex_jax_fn(
        x_ref,
        kernel_ref,
        tri_a_ref,
        tri_b_ref,
        out_conv_ref,
        out_fft_ref,
        out_tri_ref,
        out_sort_ref,
    ):
      lut = jnp.array(lut_np)
      # Indexing into a constant table + elementwise
      x = x_ref[...] + lut[None, None, : x_ref.shape[-1]]
      # cuDNN 1D convolution (N, C, W)
      conv = jax.lax.conv_general_dilated(
          x,
          kernel_ref[...],
          window_strides=(1,),
          padding="SAME",
          dimension_numbers=("NCW", "OIW", "NCW"),
      )
      out_conv_ref[...] = conv
      # cuFFT
      fft_val = jnp.fft.fft(x.astype(jnp.complex64))
      out_fft_ref[...] = jnp.real(fft_val)
      # Triangular solve (cuSOLVER / TriangularSolveThunk)
      tri_sol = jax.scipy.linalg.solve_triangular(
          tri_a_ref[...], tri_b_ref[...], lower=True
      )
      out_tri_ref[...] = tri_sol
      # Sort / CustomCall with comparator computation
      out_sort_ref[...] = jnp.sort(x, axis=-1)

    @thunky.jit
    def thunky_prog(
        x_buf,
        kernel_buf,
        tri_a_buf,
        tri_b_buf,
        out_conv_buf,
        out_fft_buf,
        out_tri_buf,
        out_sort_buf,
    ):
      # Pass a sliced sub-buffer for out_sort_buf to verify non-zero base slice rebasing
      thunky.call_jax(
          complex_jax_fn,
          x_buf,
          kernel_buf,
          tri_a_buf,
          tri_b_buf,
          out_conv_buf,
          out_fft_buf,
          out_tri_buf,
          out_sort_buf.at[1:3],
      )

    @jax.jit
    def run_all(x, kernel, tri_a, tri_b):
      out_conv = jax.new_ref(jnp.zeros_like(x))
      out_fft = jax.new_ref(jnp.zeros_like(x))
      out_tri = jax.new_ref(jnp.zeros_like(tri_b))
      out_sort_padded = jax.new_ref(
          jnp.zeros((4, x.shape[1], x.shape[2]), dtype=x.dtype)
      )
      thunky_prog(
          x,
          kernel,
          tri_a,
          tri_b,
          out_conv,
          out_fft,
          out_tri,
          out_sort_padded,
      )
      return out_conv[...], out_fft[...], out_tri[...], out_sort_padded[...]

    x_np = np.random.RandomState(0).randn(2, 4, 16).astype(np.float32)
    kernel_np = np.random.RandomState(1).randn(4, 4, 3).astype(np.float32)
    tri_a_np = (
        np.tril(np.random.RandomState(2).randn(8, 8).astype(np.float32))
        + np.eye(8, dtype=np.float32) * 3.0
    )
    tri_b_np = np.random.RandomState(3).randn(8, 4).astype(np.float32)

    res_conv, res_fft, res_tri, res_sort_padded = run_all(
        jnp.array(x_np),
        jnp.array(kernel_np),
        jnp.array(tri_a_np),
        jnp.array(tri_b_np),
    )

    x_plus_lut = x_np + lut_np[None, None, :16]
    expected_conv = jax.lax.conv_general_dilated(
        jnp.array(x_plus_lut),
        jnp.array(kernel_np),
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NCW", "OIW", "NCW"),
    )
    expected_fft = np.real(np.fft.fft(x_plus_lut.astype(np.complex64)))
    expected_tri = jax.scipy.linalg.solve_triangular(
        jnp.array(tri_a_np), jnp.array(tri_b_np), lower=True
    )
    expected_sort = np.sort(x_plus_lut, axis=-1)

    np.testing.assert_allclose(
        np.asarray(res_conv), np.asarray(expected_conv), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(res_fft), expected_fft, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(res_tri), np.asarray(expected_tri), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(res_sort_padded[1:3]), expected_sort, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
