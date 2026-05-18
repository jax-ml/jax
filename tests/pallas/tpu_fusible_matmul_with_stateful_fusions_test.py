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

"""Fusible matmul test."""

import enum
import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

jit_no_excess_precision = functools.partial(
    jax.jit, compiler_options={'xla_allow_excess_precision': False}
)


def mm_ref(x, y):
  return jnp.dot(x, y, preferred_element_type=jnp.float32)


def matmul_kernel(
    x_scalar_prefetch,
    y_scalar_prefetch,
    z_scalar_prefetch,
    x_value_refs,
    y_value_refs,
    z_value_refs,
    o_ref,
    acc_ref,
    *,
    x_fn: Any,
    y_fn: Any,
    z_fn: Any,
    out_dtype: jnp.dtype,
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  pids = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

  x_values = jax.tree.map(lambda ref: ref.get(), x_value_refs)
  x = x_fn(pids, scalar_prefetch, x_values)
  y_values = jax.tree.map(lambda ref: ref.get(), y_value_refs)
  y = y_fn(pids, scalar_prefetch, y_values)
  acc_ref[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

  @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
  def _():
    acc = acc_ref[...].astype(out_dtype)
    z_values = jax.tree.map(lambda ref: ref.get(), z_value_refs)
    out = z_fn(pids, scalar_prefetch, z_values, acc)
    jax.tree.map(lambda ref, x: ref.set(x), o_ref, out)


class KernelImpl(enum.Enum):
  PALLAS_CALL = enum.auto()
  CORE_MAP = enum.auto()
  KERNEL = enum.auto()


def _fusible_matmul(
    x: fuser.Fusion[[], jax.Array],
    y: fuser.Fusion[[], jax.Array],
    z: fuser.Fusion[[jax.Array], jax.Array] | None,
    *,
    bm: int,
    bk: int,
    bn: int,
    interpret: bool,
    debug: bool,
    impl: KernelImpl,
) -> jax.Array:
  m, k = x.shape
  k_, n = y.shape
  out_dtype = jnp.float32
  z_type = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
  if not z:
    z = lambda x: x
  if k != k_:
    raise ValueError(f'X and Y shapes must be compatible. Got {k} != {k_}')

  assert m % bm == 0
  assert k % bk == 0
  assert n % bn == 0
  grid = (m // bm, n // bn, k // bk)

  def x_index_map(i, j, k, *_):
    del j
    return i, k

  x_block_spec = pl.BlockSpec(block_shape=(bm, bk), index_map=x_index_map)

  def y_index_map(i, j, k, *_):
    del i
    return k, j

  y_block_spec = pl.BlockSpec(block_shape=(bk, bn), index_map=y_index_map)

  def z_index_map(i, j, k, *_):
    del k
    return i, j

  z_block_spec = pl.BlockSpec(block_shape=(bm, bn), index_map=z_index_map)
  dimension_semantics = (pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)

  # First thing we do is extract the values from the fusions. These will be
  # values that are passed in directly and values that are passed in via
  # scalar prefetch.
  #
  # NOTE: If the fusions read and write from Refs, then `x_values`, `y_values`,
  # and `z_values` will contain Refs.  But `x_fn`, `y_fn`, and `z_fn` will be
  # pure versions of the fusions -- i.e., taking array arguments instead of
  # Refs.  `z_fn` will return additional outputs with the updated values for
  # any Refs modified by the original, stateful fusions, and we will use
  # `z_output_input_aliases` to ensure these updates are written to the
  # correct Refs.
  x_fn, x_values, x_scalar_prefetch = fuser.get_stateful_input_fusion_values(x)
  y_fn, y_values, y_scalar_prefetch = fuser.get_stateful_input_fusion_values(y)
  z_fn, z_values, z_scalar_prefetch, z_output_input_aliases = (
      fuser.get_stateful_output_fusion_values(z, z_type))

  # Check if either input fusion reads from a Ref that is also written by the
  # output fusion.  We forbid this, as the kernel would give unexpected results
  # that depend on the exact order in which we visit blocks (and on details of
  # the pipelining).
  written_refs = [z_values[o] for o in z_output_input_aliases.keys()]
  if any(any(v is w for v in x_values + y_values) for w in written_refs):
    raise ValueError('fusible_matmul does not support an input fusion reading '
                     'from a Ref that is also written by the output fusion')

  def types_without_refs(values):
    return jax.tree.map(
        lambda v: v.inner_aval if isinstance(v, jax.ref.AbstractRef) else v,
        jax.tree.map(jax.typeof, values))

  z_out_type = jax.eval_shape(z_fn, types_without_refs(z_values), z_type)

  # We construct the set of scalar prefetch arguments that will be passed to
  # the kernel.
  scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

  x_fn, (x_value_block_specs,), _ = fuser.pull_block_spec(
      x_fn,
      x_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
      grid_len=len(grid),
  )(types_without_refs(x_values))

  y_fn, (y_value_block_specs,), _ = fuser.pull_block_spec(
      y_fn,
      y_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(1),
      grid_len=len(grid),
  )(types_without_refs(y_values))

  z_out_block_spec = fuser.push_block_spec(
      z_fn, tuple([pl.no_block_spec] * len(z_values)), z_block_spec
  )(types_without_refs(z_values), z_type)
  z_fn, (z_value_block_specs, _), _ = fuser.pull_block_spec(
      z_fn,
      z_out_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(2),
      grid_len=len(grid),
  )(types_without_refs(z_values), z_type)

  # TODO(sharadmv): This is a hack. We should be able to pass in the scalar
  # prefetch arguments directly to the kernel but don't have Mosaic support atm.
  scalar_prefetch = jax.tree.map(lambda x: x[None], scalar_prefetch)

  if impl == KernelImpl.PALLAS_CALL:
    input_shift = len(jax.tree.leaves((scalar_prefetch, x_values, y_values)))
    input_output_aliases = {
        input_shift + i: o for o, i in z_output_input_aliases.items()
    }

    def get(x):
      if isinstance(x, jax.ref.Ref):
        return x[...]
      return x

    out = pl.pallas_call(
        functools.partial(
            matmul_kernel,
            x_fn=x_fn,
            y_fn=y_fn,
            z_fn=z_fn,
            out_dtype=out_dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetch),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            in_specs=[
                x_value_block_specs,
                y_value_block_specs,
                z_value_block_specs,
            ],
            out_specs=[z_out_block_spec],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=dimension_semantics,
        ),
        out_shape=[z_out_type],
        input_output_aliases=input_output_aliases,
        interpret=interpret,
        debug=debug,
    )(*scalar_prefetch, *jax.tree.map(get, (x_values, y_values, z_values)))[0]

    if z_output_input_aliases:
      flat_out = jax.tree.leaves(out)
      for o, i in z_output_input_aliases.items():
        z_values[i][...] = flat_out[o]
      return tuple(v for o, v in enumerate(flat_out)
                   if o not in z_output_input_aliases)
    return out

  elif impl == KernelImpl.CORE_MAP:
    z_out_leaves, z_out_tree = jax.tree.flatten(z_out_type)
    out = z_out_tree.unflatten(
        z_values[z_output_input_aliases[i]] if i in z_output_input_aliases
        else jnp.empty_like(v)
        for i, v in enumerate(z_out_leaves))

    def ref(x):
      if isinstance(x, jax.ref.Ref):
        return x
      return jax.new_ref(x)

    (x_values_refs, y_values_refs, z_values_refs, scalar_prefetch_refs, out_ref
     ) = jax.tree.map(ref, (x_values, y_values, z_values, scalar_prefetch, out))

    @pl.core_map(pltpu.create_tensorcore_mesh('core'),
                  interpret=interpret, debug=debug)
    def _():
      def _f(acc_ref, scalar_prefetch_smem_refs):
        pltpu.sync_copy(scalar_prefetch_refs, scalar_prefetch_smem_refs)

        def block_spec_with_prefetch(bs):
          if bs is pl.no_block_spec:
            return pl.BlockSpec()
          if bs.index_map is None:
            return bs
          return bs.replace(
            index_map=lambda *args: bs.index_map(
                *args, *scalar_prefetch_smem_refs))

        in_specs_ = jax.tree.map(block_spec_with_prefetch, (
            x_value_block_specs, y_value_block_specs, z_value_block_specs))
        z_out_block_spec_ = jax.tree.map(
            block_spec_with_prefetch, z_out_block_spec)
        pltpu.emit_pipeline(
            functools.partial(
                    matmul_kernel,
                    *scalar_prefetch_smem_refs,
                    acc_ref=acc_ref,
                    x_fn=x_fn,
                    y_fn=y_fn,
                    z_fn=z_fn,
                    out_dtype=out_dtype),
            grid=grid,
            in_specs=in_specs_,
            out_specs=[z_out_block_spec_],
            core_axis_name='core',
            dimension_semantics=dimension_semantics,
        )(x_values_refs, y_values_refs, z_values_refs, out_ref)
      pl.run_scoped(_f,
                    pltpu.VMEM((bm, bn), jnp.float32),
                    jax.tree.map(pltpu.SMEM.like, scalar_prefetch))

    out = tuple(r.get() for o, r in enumerate(jax.tree.leaves(out_ref))
                if o not in z_output_input_aliases)
    if z_output_input_aliases:
      return out
    else:
      return z_out_tree.unflatten(out)

  elif impl == KernelImpl.KERNEL:
    z_out_leaves, z_out_tree = jax.tree.flatten(z_out_type)
    out = z_out_tree.unflatten(
        z_values[z_output_input_aliases[i]] if i in z_output_input_aliases
        else jnp.empty_like(v)
        for i, v in enumerate(z_out_leaves))

    def ref(x):
      if isinstance(x, jax.ref.Ref):
        return x
      return jax.new_ref(x)
    # NOTE: We have to close over these values instead of passing them to
    # `pl.kernel(body, ...)` because the same Ref may appear multiple times
    # across x_values, ..., out, which is not supported in
    # `pl.kernel(...)` args.
    (x_values_refs, y_values_refs, z_values_refs, out_ref) = jax.tree.map(
        ref, (x_values, y_values, z_values, out))

    def body(scalar_prefetch_refs, acc_vmem_ref, scalar_prefetch_smem_refs):
      pltpu.sync_copy(scalar_prefetch_refs, scalar_prefetch_smem_refs)

      def block_spec_with_prefetch(bs):
        if bs is pl.no_block_spec:
          return pl.BlockSpec()
        if bs.index_map is None:
          return bs
        return bs.replace(
          index_map=lambda *args: bs.index_map(
              *args, *scalar_prefetch_smem_refs))

      in_specs_ = jax.tree.map(block_spec_with_prefetch, (
          x_value_block_specs, y_value_block_specs, z_value_block_specs))
      z_out_block_spec_ = jax.tree.map(
          block_spec_with_prefetch, z_out_block_spec)
      pltpu.emit_pipeline(
          functools.partial(
              matmul_kernel,
              *scalar_prefetch_smem_refs,
              acc_ref=acc_vmem_ref,
              x_fn=x_fn,
              y_fn=y_fn,
              z_fn=z_fn,
              out_dtype=out_dtype),
          grid=grid,
          in_specs=in_specs_,
          out_specs=[z_out_block_spec_],
          core_axis_name='core',
          dimension_semantics=dimension_semantics
      )(x_values_refs, y_values_refs, z_values_refs, out_ref)

    pl.kernel(
        body,
        mesh=pltpu.create_tensorcore_mesh('core'),
        scratch_types=[
            pltpu.VMEM((bm, bn), jnp.float32),
            jax.tree.map(pltpu.SMEM.like, scalar_prefetch),
        ],
        interpret=interpret,
        debug=debug,
    )(scalar_prefetch)

    out = tuple(r.get() for o, r in enumerate(jax.tree.leaves(out_ref))
                if o not in z_output_input_aliases)
    if z_output_input_aliases:
      return out
    else:
      return z_out_tree.unflatten(out)

  else:
    raise ValueError(f'Unsupported impl: {impl}')


def fusible_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    debug: bool = False,
    interpret: bool = False,
    impl: KernelImpl = KernelImpl.CORE_MAP,
) -> jax.Array:
  return fuser.fusible(
      functools.partial(
          _fusible_matmul,
          bm=bm,
          bk=bk,
          bn=bn,
          interpret=interpret,
          debug=debug,
          impl=impl,
      )
  )(x, y)


class FusibleMatmulTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works with TPU v4+')
    super().setUp()

  @parameterized.product(dtype=['float32', 'bfloat16'], impl=list(KernelImpl))
  def test_matmul_with_bias_and_read_only_refs(self, dtype, impl):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)
    a = jax.random.normal(k2, (512, 512), dtype)
    b = jax.random.normal(k3, (512, 512), dtype)

    @jit_no_excess_precision(static_argnames=['fuse'])
    def run_matmul(x, y, a, b, fuse=True):
      # NOTE: Currently, fused functions (and kernels) can only close over Refs
      # that were created inside of jax.jit.
      a_ref = jax.new_ref(a)
      b_ref = jax.new_ref(b)

      _fusible_matmul = functools.partial(fusible_matmul, impl=impl)
      if not fuse:
        _fusible_matmul = fuser.fuse(_fusible_matmul)

      def matmul(x, y):
        x = _fusible_matmul(x + a_ref[...], y + b_ref[...] * 0.5).astype(dtype)
        return jnp.maximum(x + b_ref[...], 0.0)

      if fuse:
        matmul = fuser.fuse(matmul)

      return matmul(x, y)

    @jit_no_excess_precision
    def matmul_ref(x, y, a, b):
      return jax.nn.relu(mm_ref(x + a, y + b * 0.5).astype(dtype) + b)

    self.assertArraysEqual(run_matmul(x, y, a, b, fuse=False),
                           run_matmul(x, y, a, b))

    np.testing.assert_allclose(
        run_matmul(x, y, a, b),
        matmul_ref(x, y, a, b),
        atol=5e-5 if dtype == 'float32' else 0.5,
    )

  @parameterized.product(dtype=['float32', 'bfloat16'], impl=list(KernelImpl))
  def test_matmul_with_bias_and_simple_aliased_ref(self, dtype, impl):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)
    b = jax.random.normal(k2, (512, 512), dtype)

    @jit_no_excess_precision(static_argnames=['fuse'])
    def run_matmul(x, y, b, fuse=True):
      # NOTE: Currently, fused functions (and kernels) can only close over Refs
      # that were created inside of jax.jit.
      b_ref = jax.new_ref(b)

      _fusible_matmul = functools.partial(fusible_matmul, impl=impl)
      if not fuse:
        _fusible_matmul = fuser.fuse(_fusible_matmul)

      def matmul(x, y):
        x = _fusible_matmul(x, y).astype(dtype)
        b_ref[...] = jnp.maximum(x + b_ref[...], 0.0)

      if fuse:
        matmul = fuser.fuse(matmul)

      matmul(x, y)
      return jax.ref.freeze(b_ref)

    @jit_no_excess_precision
    def matmul_ref(x, y, b):
      return jax.nn.relu(mm_ref(x, y).astype(dtype) + b)

    self.assertArraysEqual(run_matmul(x, y, b, fuse=False),
                           run_matmul(x, y, b))

    np.testing.assert_allclose(
        run_matmul(x, y, b),
        matmul_ref(x, y, b),
        atol=5e-5 if dtype == 'float32' else 0.5,
    )

  @parameterized.parameters(KernelImpl)
  def test_matmul_forbids_unsupported_ref_access(self, impl):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (512, 512), jnp.float32)
    y = jax.random.normal(k1, (512, 512), jnp.float32)
    b = jax.random.normal(k2, (512, 512), jnp.float32)

    @jax.jit
    def run_matmul(x, y, b):
      # NOTE: Currently, fused functions (and kernels) can only close over Refs
      # that were created inside of jax.jit.
      x_ref = jax.new_ref(x)
      b_ref = jax.new_ref(b)

      @fuser.fuse
      def matmul(x, y):
        # fusible_matmul will reject this for reading x_ref in the input fusion
        # while also writing to x_ref in the output fusion.  This computation
        # would give unexpected results that would depend on the order in which
        # the kernel reads from and writes to different blocks of x_ref.
        x_ref[...] = fusible_matmul(jnp.transpose(x_ref[...]) + b_ref[...], y)

      matmul(x, y)
      return jax.ref.freeze(b_ref)

    with self.assertRaisesRegex(Exception, 'fusible_matmul does not support'):
      run_matmul(x, y, b)

  @parameterized.parameters(KernelImpl)
  def test_matmul_with_writes_that_do_not_depend_on_output(self, impl):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (512, 512), jnp.float32)
    y = jax.random.normal(k1, (512, 512), jnp.float32)

    @jax.jit
    def run_matmul(x, y):
      # NOTE: Currently, fused functions (and kernels) can only close over Refs
      # that were created inside of jax.jit.
      side_output_ref = jax.new_ref(jnp.empty_like(x))

      @fuser.fuse
      def matmul(x, y):
        # NOTE: This write is not allowed becuase it does not depend on the
        # output of fusible_matmul.
        side_output_ref[...] = x * 2.0
        return fusible_matmul(x, y)

      res = matmul(x, y)
      return res, jax.ref.freeze(side_output_ref)

    with self.assertRaisesRegex(Exception, 'must depend on an output'):
      run_matmul(x, y)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
