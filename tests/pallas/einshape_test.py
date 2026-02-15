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

import functools
import math
import string

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as st
import jax
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


jax.config.parse_flags_with_absl()
jtu.setup_hypothesis()


@jax.jit(static_argnames=["equation", "sizes"])
def _einshape_kernel(x, equation, sizes):
  sizes = dict(sizes)
  x_ref = jax.new_ref(x)
  fn = functools.partial(pltpu.einshape, equation, **sizes)
  out_ref = jax.new_ref(jnp.empty_like(jax.eval_shape(fn, x)))

  @pl.core_map(mesh=pltpu.create_tensorcore_mesh(num_cores=1, axis_name="x"))
  def _():
    @pl.with_scoped(
        pltpu.VMEM(x_ref.shape, x_ref.dtype),
        pltpu.VMEM(out_ref.shape, out_ref.dtype),
    )
    def inner(x_vmem_ref, out_vmem_ref):
      pltpu.sync_copy(x_ref, x_vmem_ref)
      out_vmem_ref[...] = pltpu.einshape(equation, x_vmem_ref[...], **sizes)
      pltpu.sync_copy(out_vmem_ref, out_ref)
    inner()

  return out_ref[...]


@st.composite
def einshape_strategy(draw, dtype: jnp.dtype, has_shape_constraint: bool):

  def partition(lst):
    # Partitions a list using hypothesis draws
    if not lst:
      return []
    groups = []
    i = 0
    while i < len(lst):
      g_size = draw(st.integers(1, len(lst) - i))
      groups.append(lst[i : i + g_size])
      i += g_size
    return groups

  if has_shape_constraint:
    # Construct strictly tile-preserving configurations
    num_outer = draw(st.integers(0, 3))
    outer_names = list(string.ascii_lowercase[:num_outer])
    s_name = string.ascii_lowercase[num_outer + 1]
    l_name = string.ascii_lowercase[num_outer + 2]

    dim_sizes = {n: draw(st.integers(1, 3)) for n in outer_names}
    p = 32 // dtypes.itemsize_bits(dtype)
    # sublane dimensions
    dim_sizes[s_name] = draw(st.sampled_from([8 * p, 16 * p, 32 * p]))
    # lane dimension
    dim_sizes[l_name] = draw(st.sampled_from([128, 256, 512]))

    def _generate_groups(names):
      # Slice names into 3 random groups. One will be used for leading
      # dimensions and the other two will be folded into the sublane and lane
      # dimensions respectively.
      # E.g. if _names = [a, b, c, d] and we draw idx1=1, idx2=3, then
      # free = [a], s = [b, c], l = [d]
      idx1 = draw(st.integers(0, len(names)))
      idx2 = draw(st.integers(idx1, len(names)))
      free, s, l = names[:idx1], names[idx1:idx2], names[idx2:]
      # The free group is split up further randomly.
      groups = partition(free) + [
          s + [s_name],
          l + [l_name],
      ]
      return groups
    lhs_groups = _generate_groups(outer_names)
    # Only "outer" dimensions are permuted.
    rhs_outer_names = list(draw(st.permutations(outer_names)))
    rhs_groups = _generate_groups(rhs_outer_names)
  else:
    num_dims = draw(st.integers(1, 6))
    all_names = list(string.ascii_lowercase[:num_dims])
    dim_sizes = {n: draw(st.integers(1, 4)) for n in all_names}

    lhs_flat_names = all_names
    lhs_groups = partition(lhs_flat_names)

    rhs_flat_names = list(draw(st.permutations(all_names)))
    rhs_groups = partition(rhs_flat_names)

  def format_side(groups):
    return "".join(g[0] if len(g) == 1 else f"({''.join(g)})" for g in groups)

  equation = f"{format_side(lhs_groups)}->{format_side(rhs_groups)}"

  kwargs = {}
  for g in lhs_groups:
    if len(g) > 1:
      drop_idx = draw(st.integers(0, len(g) - 1))
      for i, n in enumerate(g):
        if i != drop_idx:
          kwargs[n] = dim_sizes[n]

  lhs_shape = tuple(math.prod(dim_sizes[n] for n in g) for g in lhs_groups)
  rhs_shape = tuple(math.prod(dim_sizes[n] for n in g) for g in rhs_groups)

  lhs_flat = [n for g in lhs_groups for n in g]
  rhs_flat = [n for g in rhs_groups for n in g]
  atomic_shape = tuple(dim_sizes[n] for n in lhs_flat)
  perm = tuple(lhs_flat.index(n) for n in rhs_flat)

  return {
      "equation": equation,
      "lhs_shape": lhs_shape,
      "rhs_shape": rhs_shape,
      "kwargs": kwargs,
      "atomic_shape": atomic_shape,
      "perm": perm,
  }


class EinshapeTest(jtu.JaxTestCase):

  has_shape_constraint: bool = False

  def impl(self, equation, x, **sizes):
    return pltpu.einshape(equation, x, **sizes)

  @parameterized.product(
      einshape=[
          # TODO(sharadmv): why does this test time out? Mosaic padding bug?
          # ("a(bc)->abc", (2, 3 * 128), {"c": 128}),
          ("ab(cd)->cabd", (2, 4, 128 * 4), {"c": 4}),
          ("abcd->ab(cd)", (2, 3, 4, 128), {}),
          ("abc->a(bc)", (8, 2, 128), {}),
          ("a(bc)->abc", (10, 128 * 4), {"b": 4}),
          ("a(bc)->abc", (10, 128 * 2), {"c": 128}),
          ("a(bc)->abc", (10, 128 * 2), {"b": 2}),
          ("a(bc)->b(ac)", (8, 256), {"c": 128}),
      ],
      dtype=["int32", "bfloat16"],
  )
  def test_einshape_basic(self, einshape, dtype):
    equation, shape, sizes = einshape
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(dtype)
    out = self.impl(equation, x, **sizes)

    match equation:
      case "ab->ab":
        expected = x
      case "abc->(ab)c":
        expected = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
      case "ab(cd)->cabd":
        a, b, cd = x.shape
        c = sizes["c"]
        d = cd // c
        expected = x.reshape(a, b, c, d).transpose(2, 0, 1, 3)
      case "abcd->ab(cd)":
        a, b, c, d = x.shape
        expected = x.reshape(a, b, c * d)
      case "abc->a(bc)":
        a, b, c = x.shape
        expected = x.reshape(a, b * c)
      case "a(bc)->abc":
        a, bc = x.shape
        if "b" in sizes:
          b = sizes["b"]
          c = bc // b
        else:
          c = sizes["c"]
          b = bc // c
        expected = x.reshape(a, b, c)
      case "a(bc)->b(ac)":
        a, bc = x.shape
        c = sizes["c"]
        b = bc // c
        expected = x.reshape(a, b, c).transpose(1, 0, 2).reshape(b, a * c)
      case _:
        raise ValueError(f"Unsupported equation: {equation}")
    self.assertArraysEqual(out, expected)

  def test_error_ambiguous(self):
    x = jnp.zeros((10, 12))
    with self.assertRaisesRegex(ValueError, "Ambiguous split"):
      self.impl("a(bc)->abc", x)

  def test_error_mismatch(self):
    x = jnp.zeros((10, 13))
    with self.assertRaisesRegex(ValueError, "Cannot split size"):
      self.impl("a(bc)->abc", x, b=3)

  @hp.given(data=st.data(), dtype=st.sampled_from(["int32", "bfloat16"]))
  @hp.settings(max_examples=200)
  def test_hypothesis_einshape(self, data, dtype):
    case = data.draw(einshape_strategy(dtype, self.has_shape_constraint))

    equation = case["equation"]
    lhs_shape = case["lhs_shape"]
    rhs_shape = case["rhs_shape"]
    kwargs = case["kwargs"]
    atomic_shape = case["atomic_shape"]
    perm = case["perm"]

    x = jnp.arange(math.prod(lhs_shape)).reshape(lhs_shape).astype(dtype)
    out = self.impl(equation, x, **kwargs)

    self.assertEqual(out.shape, rhs_shape)

    x_atomic = x.reshape(atomic_shape)
    x_transposed = jax.lax.transpose(x_atomic, perm)
    expected = x_transposed.reshape(rhs_shape)

    self.assertArraysEqual(out, expected)


class EinshapeTPUKernelTest(EinshapeTest):

  has_shape_constraint: bool = True

  def impl(self, equation, x, **sizes):
    return _einshape_kernel(x, equation, tuple(sizes.items()))

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Skipping test because TPU is not supported.")


if __name__ == "__main__":
  absltest.main()
