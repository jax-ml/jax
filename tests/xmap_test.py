# Copyright 2020 Google LLC
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

# flake8: noqa

from contextlib import contextmanager
import functools
import itertools as it
import os
import unittest
from itertools import product, permutations
from typing import (Tuple, List, NamedTuple, Dict, Generator, Sequence, Set,
                    Any, Hashable, Iterable, Iterator, Union)
from unittest import SkipTest, skip, skipIf

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from functools import partial

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax import vmap
from jax import lax
from jax.experimental.maps import Mesh, mesh, xmap
from jax.lib import xla_bridge
from jax._src.util import curry, unzip2, split_list, prod
from jax._src.lax.lax import DotDimensionNumbers
from jax.interpreters import pxla

from jax.config import config
config.parse_flags_with_absl()

ignore_xmap_warning = functools.partial(
  jtu.ignore_warning, message=".*is an experimental.*")

# TODO(mattjj): de-duplicate setUpModule and tearDownModule with pmap_test.py
# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()

MeshSpec = List[Tuple[str, int]]

@contextmanager
def with_mesh(named_shape: MeshSpec) -> Generator[None, None, None]:
  """Test utility for setting up meshes given mesh data from `schedules`."""
  # This is similar to the `with_mesh` function above, but isn't a decorator.
  axis_names, shape = unzip2(named_shape)
  size = prod(shape)
  local_devices = list(jax.local_devices())
  if len(local_devices) < size:
    raise SkipTest(f"Test requires {size} local devices")
  mesh_devices = np.array(local_devices[:size]).reshape(shape)
  with mesh(mesh_devices, axis_names):
    yield


class XMapTest(jtu.JaxTestCase):
  def setUp(self):
    if jax.lib.version < (0, 1, 58):
      raise SkipTest("xmap requires jaxlib version >= 0.1.58")
    if not config.omnistaging_enabled:
      raise SkipTest("xmap requires omnistaging")

  @ignore_xmap_warning()
  def testBasic(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return a * 2, b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[{0: 'a', 1: 'b'}, ['c', ...]],
                out_axes=[{0: 'a', 1: 'b'}, ['c', ...]],
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, a * 2)
      self.assertAllClose(d, b * 4)

  @ignore_xmap_warning()
  def testBasicCollective(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return lax.psum(a * 2, 'a'), b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[['a', 'b', ...], {0: 'c'}],
                out_axes=[['b', ...], {0: 'c'}],
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, (a * 2).sum(0))
      self.assertAllClose(d, b * 4)

  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesBasic(self):
    def f(v):
      return lax.psum(v * 2, 'a'), v * 4
    fm = xmap(f, in_axes=['a', ...], out_axes=[{}, {1: 'a'}],
              axis_resources={'a': ('x', 'y')})
    vshape = (4, 5)
    v = jnp.arange(np.prod(vshape)).reshape(vshape)
    ans, ans2 = fm(v)
    self.assertAllClose(ans, (v * 2).sum(0))
    self.assertAllClose(ans2, v.T * 4)

  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesSharding(self):
    def f(v):
      return v * 4
    fxy = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('x', 'y')})
    fyx = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('y', 'x')})
    vshape = (4, 5)
    v = jnp.arange(np.prod(vshape)).reshape(vshape)
    zxy = fxy(v)
    self.assertEqual(
        zxy.sharding_spec,
        pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked((2, 2))),
                          (pxla.ShardedAxis(0), pxla.ShardedAxis(1))))
    zyx = fyx(v)
    self.assertEqual(
        zyx.sharding_spec,
        pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked((2, 2))),
                          (pxla.ShardedAxis(1), pxla.ShardedAxis(0))))

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testCompilationCache(self):
    def f(x):
      assert python_should_be_executing
      return x * 2
    fm = xmap(f,
              in_axes=['a', ...], out_axes=['a', ...],
              axis_resources={'a': 'x'})
    x = np.arange(8).reshape((2, 2, 2))
    python_should_be_executing = True
    fm(x)
    python_should_be_executing = False
    fm(x)

  @skip("Need to implement vmap(xmap)")
  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 3)])
  def testNestedMesh(self):
    @partial(xmap, in_axes={1: 'a'}, out_axes={0: 'a'}, axis_resources={'a': 'y'})
    def f(x):
      y = x * 2
      @partial(xmap, in_axes={0: 'b'}, out_axes={1: 'b'}, axis_resources={'b': 'x'})
      def h(y):
        return jnp.sin(y)
      return h(y)
    xshape = (2, 3, 5)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    y = f(x)
    self.assertAllClose(y, jnp.sin(x * 2).transpose((1, 2, 0)))
    # Make sure the op really ran accros a 2D mesh.
    self.assertEqual(y.sharding_spec.sharding,
                     (pxla.Chunked(3), None, None))
    self.assertEqual(y.sharding_spec.mesh_mapping,
                     (pxla.Replicated(2), pxla.ShardedAxis(0)))

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testNestedDifferentResources(self):
    @partial(xmap, in_axes={0: 'a'}, out_axes={0: 'a'}, axis_resources={'a': 'x'})
    def f(x):
      with mesh(np.empty((), dtype=np.object), ()):
        @partial(xmap, in_axes={0: 'b'}, out_axes={0: 'b'})
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(RuntimeError,
                                "Changing the resource environment.*"):
      f(x)

  @parameterized.named_parameters(
    {"testcase_name": name, "mesh": mesh, "axis_resources": axis_resources}
    for name, mesh, axis_resources in (
      ('', (), ()),
      ('Mesh', (('x', 2),), (('i', 'x'),))
    ))
  @ignore_xmap_warning()
  def testMultipleCalls(self, mesh, axis_resources):
    def f(x, y):
      assert x.shape == y.shape == (3, 5)
      return jnp.tensordot(x, y, axes=([1], [1]))

    @with_mesh(mesh)
    def run_test():
      f_mapped = xmap(f,
                      in_axes=(['i', ...], ['j', ...]),
                      out_axes=['i', 'j', ...],
                      axis_resources=dict(axis_resources))
      x = jnp.arange(30).reshape(2, 3, 5)
      expected = jnp.einsum('imk,jnk->ijmn', x, x)
      for i in range(10):
        self.assertAllClose(f_mapped(x, x), expected)
    run_test()

  def VmapOfXmapCases():
    xmap_in_axes = ([{}] +
                    [{i: 'x'} for i in range(3)] +
                    [{i: 'x', j: 'y'} for i in range(4) for j in range(4) if i != j])
    for xmap_dim_x, xmap_dim_y in product(xmap_in_axes, repeat=2):
      xmap_axes = sorted(set(xmap_dim_x.values()) | set(xmap_dim_y.values()))
      num_axes = len(xmap_axes)
      if xmap_axes is None:
        continue
      xmap_out_axes = [dict(zip(dims, xmap_axes))
                       for dims in permutations(range(2 + num_axes), num_axes)]
      for xmap_dim_z in xmap_out_axes:
        for vmap_dim_x in [*range(2 + len(xmap_dim_x)), None]:
          for vmap_dim_y in [*range(2 + len(xmap_dim_y)), None]:
            if vmap_dim_x is None and vmap_dim_y is None:
              continue
            for vmap_dim_z in range(2 + len(xmap_axes)):
              for vmap_as_xmap in [False, True]:
                yield {"testcase_name":
                          f"_xin={(sorted(xmap_dim_x.items()), sorted(xmap_dim_y.items()))}_"
                          f"xout={sorted(xmap_dim_z.items())}_vin={(vmap_dim_x, vmap_dim_y)}_"
                          f"vout={vmap_dim_z}_vmap_as_xmap={vmap_as_xmap}",
                       "xmap_in_axes": (xmap_dim_x, xmap_dim_y),
                       "xmap_out_axes": xmap_dim_z,
                       "vmap_in_axes": (vmap_dim_x, vmap_dim_y),
                       "vmap_out_axes": vmap_dim_z,
                       "vmap_as_xmap": vmap_as_xmap}

  @parameterized.named_parameters(jtu.cases_from_list(VmapOfXmapCases()))
  @ignore_xmap_warning()
  def testNestedMap(self, xmap_in_axes, xmap_out_axes, vmap_in_axes, vmap_out_axes, vmap_as_xmap):
    """Test various vmap(xmap) and xmap(xmap) combinations.

    The outer map always introduces a single dimension, the inner map introduces one or two.
    """
    (xin_x, xin_y) = xmap_in_axes
    (vin_x, vin_y) = vmap_in_axes
    vmap_size = 7
    xmap_sizes = {'x': 11, 'y': 13}

    xshape = [2, 3]
    yshape = [3, 5]
    zshape = [2, 5]
    xind = ['n', 'k']
    yind = ['k', 'm']
    zind = ['n', 'm']
    f = partial(jnp.einsum, 'nk,km->nm')

    for pos, name in sorted(xin_x.items()):
      xshape.insert(pos, xmap_sizes[name])
      xind.insert(pos, name)
    for pos, name in sorted(xin_y.items()):
      yshape.insert(pos, xmap_sizes[name])
      yind.insert(pos, name)
    for pos, name in sorted(xmap_out_axes.items()):
      zshape.insert(pos, xmap_sizes[name])
      zind.insert(pos, name)

    if vin_x is not None:
      xshape.insert(vin_x, vmap_size)
      xind.insert(vin_x, 'v')
    if vin_y is not None:
      yshape.insert(vin_y, vmap_size)
      yind.insert(vin_y, 'v')
    zshape.insert(vmap_out_axes, vmap_size)
    zind.insert(vmap_out_axes, 'v')

    if vmap_as_xmap:
      do_vmap = partial(xmap,
                        in_axes=({vin_x: 'v'} if vin_x is not None else {},
                                 {vin_y: 'v'} if vin_y is not None else {}),
                        out_axes={vmap_out_axes: 'v'})
    else:
      do_vmap = partial(vmap, in_axes=vmap_in_axes, out_axes=vmap_out_axes)

    fm = do_vmap(xmap(f, in_axes=xmap_in_axes, out_axes=xmap_out_axes))
    fref = partial(jnp.einsum, f"{''.join(xind)},{''.join(yind)}->{''.join(zind)}")

    rng = np.random.RandomState(0)
    x = rng.randn(*xshape)
    y = rng.randn(*yshape)
    self.assertAllClose(fm(x, y), fref(x, y))


class XMapTestSPMD(XMapTest):
  """Re-executes all tests with the SPMD partitioner enabled"""

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != "tpu":
      raise SkipTest
    jax.experimental.maps.make_xmap_callable.cache_clear()
    self.old_lowering_flag = jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING
    jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = True

  def tearDown(self):
    jax.experimental.maps.make_xmap_callable.cache_clear()
    jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = self.old_lowering_flag

AxisIndices = Tuple[int, ...]
MatchedAxisIndices = Tuple[AxisIndices, AxisIndices]
AxisNames = Tuple[str, ...]

class PdotTestSpec:
  # The axis indices stored by a PdotTestSpec are all positional indices
  # *before* taking mapping into account.
  map_cont: MatchedAxisIndices
  pos_cont: MatchedAxisIndices
  map_batch: MatchedAxisIndices
  pos_batch: MatchedAxisIndices
  all_names: AxisNames
  contract_names: AxisNames
  batch_names: AxisNames

  def __init__(self, map_cont, pos_cont, map_batch, pos_batch):
    self.map_cont = map_cont
    self.pos_cont = pos_cont
    self.map_batch = map_batch
    self.pos_batch = pos_batch

    names = gen_axis_names()
    self.contract_names = [next(names) for _ in range(len(map_cont[0]))]
    self.batch_names = [next(names) for _ in range(len(map_batch[0]))]
    self.all_names = self.contract_names + self.batch_names

  @property
  def dot_general_dim_nums(self):
    lhs_contract = (*self.map_cont[0], *self.pos_cont[0])
    rhs_contract = (*self.map_cont[1], *self.pos_cont[1])
    lhs_batch = (*self.map_batch[0], *self.pos_batch[0])
    rhs_batch = (*self.map_batch[1], *self.pos_batch[1])
    return (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

  @property
  def pos_contract_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_cont[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_cont[1]]
    return (lhs, rhs)

  @property
  def pos_batch_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_batch[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_batch[1]]
    return (lhs, rhs)

  @property
  def _lhs_mapped(self):
    return {*self.map_cont[0], *self.map_batch[0]}

  @property
  def _rhs_mapped(self):
    return {*self.map_cont[1], *self.map_batch[1]}

  @property
  def lhs_in_axes(self):
    axis_indices = [*self.map_cont[0], *self.map_batch[0]]
    return dict(zip(axis_indices, self.all_names))

  @property
  def rhs_in_axes(self):
    axis_indices = [*self.map_cont[1], *self.map_batch[1]]
    return dict(zip(axis_indices, self.all_names))

def all_pdot_specs(lhs_shape, rhs_shape):
  for matching in axis_matchings(lhs_shape, rhs_shape):
    for lists in partitions(matching, 4):
      yield PdotTestSpec(*map(unzip2, lists))

def axis_matchings(lhs_shape, rhs_shape):
  def helper(start, exc1, exc2):
    yield ()
    for i in range(start, len(lhs_shape)):
      d1 = lhs_shape[i]
      if i not in exc1:
        for j, d2 in enumerate(rhs_shape):
          if d1 == d2 and j not in exc2:
            for matches in helper(i + 1, exc1 | {i}, exc2 | {j}):
              yield ((i, j), *matches)
  return helper(0, set(), set())

def partitions(s, k):
  for indices in product(range(k), repeat=len(s)):
    outs = [[] for _ in range(k)]
    for i, elt in zip(indices, s):
      outs[i].append(elt)
    yield outs

def powerset(s):
  s = list(s)
  return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def gen_axis_names():
  names = 'ijkl'
  for n in it.count(1):
    for chars in product(names, repeat=n):
      yield ''.join(chars)

AxisResources = Dict[str, Union[str, Tuple[str, ...]]]

def schedules(sizes: Dict[str, int]
              ) -> Generator[Tuple[AxisResources, MeshSpec], None, None]:
  """Test utility generating xmap parallel schedules from logical names & sizes.

  Args:
    sizes: dict mapping logical axis name to its corresponding size.

  Returns:
    A generator producing finitely many values, where each value is a pair in
    which the first element is a value suitable for xmap's axis_resources
    argument and the second element is a list of pairs with the first element
    representing a generated physical mesh axis name and the second element
    representing a corresponding generated mesh axis size. The generated mesh
    names/sizes can be used to define a physical mesh in tests.

  This function doesn't generate schedules which map distinct logical axis names
  to the same parallel resource name. It only generates parallel resources; the
  rest are implicitly left for vectorization. Parallel resource names are
  generated by prepending an 'r', 'r1', or 'r2' to the corresponding logical
  name.

  Exa,mples:
    >>> for sched in schedules({'i': 2, 'j': 4}):
    ...   print(sched)
    ({}, [])
    ({'i': 'ri'}, [('ri', 1)])
    ({'i': 'ri'}, [('ri', 2)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 1), ('r2i', 1)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 1), ('r2i', 2)])
    ({'i': ('r1i', 'r2i')}, [('r1i', 2), ('r2i', 1)])
    ({'j': 'rj'}, [('rj', 1)])
    ({'j': 'rj'}, [('rj', 2)])
    ({'j': 'rj'}, [('rj', 4)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 1)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 2)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 1), ('r2j', 4)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 2), ('r2j', 1)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 2), ('r2j', 2)])
    ({'j': ('r1j', 'r2j')}, [('r1j', 4), ('r2j', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 2)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 1), ('rj', 4)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 1)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 2)])
    ({'i': 'ri', 'j': 'rj'}, [('ri', 2), ('rj', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 1), ('r2j', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 2), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 2), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 1), ('r1j', 4), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 1), ('r2j', 4)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 2), ('r2j', 1)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 2), ('r2j', 2)])
    ({'i': 'ri', 'j': ('r1j', 'r2j')}, [('ri', 2), ('r1j', 4), ('r2j', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 1), ('r1i', 2), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 2), ('r1i', 2), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 1), ('r2i', 1)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 1), ('r2i', 2)])
    ({'j': 'rj', 'i': ('r1i', 'r2i')}, [('rj', 4), ('r1i', 2), ('r2i', 1)])
  """
  def divisors(n: int) -> List[int]:
    return [m for m in range(1, n + 1) if not n % m]

  def divisors2(n: int) -> Iterator[Tuple[int, int]]:
    for k1 in divisors(n):
      for k2 in divisors(n // k1):
        yield (k1, k2)

  # choose a subset of logical axis names to map to parallel resources
  for names in powerset(sizes):
    # partition that set of logical axis names into two subsets: one subset to
    # map to one parallel resource axis and a second subset to map to two
    # parallel resource axes.
    for names1, names2 in partitions(names, 2):
      # to avoid generating too many complex cases, we skip generating cases
      # where more than one logical axis name is to be mapped to two parallel
      # resource axes. comment out this line to generate more complex tests.
      if len(names2) > 1: continue
      # make up parallel resource axis names for each logical axis
      axis_resources1 = ((name, 'r' + name) for name in names1)
      axis_resources2 = ((name, ('r1' + name, 'r2' + name)) for name in names2)
      axis_resources = dict(it.chain(axis_resources1, axis_resources2))
      # make up sizes for each resource axis, where the size must divide the
      # corresponding logical axis
      for mesh_sizes1 in product(*(divisors(sizes[n]) for n in names1)):
        for mesh_sizes2 in product(*(divisors2(sizes[n]) for n in names2)):
          mesh_data1 = (('r' + name, size) for name, size in zip(names1, mesh_sizes1))
          mesh_data2 = (pair for name, (size1, size2) in zip(names2, mesh_sizes2)
                        for pair in [('r1' + name, size1), ('r2' + name, size2)])
          mesh_data = list(it.chain(mesh_data1, mesh_data2))
          yield axis_resources, mesh_data

def schedules_from_pdot_spec(
    spec: PdotTestSpec, lhs_shape: Tuple[int], rhs_shape: Tuple[int]
    ) -> Generator[Tuple[AxisResources, MeshSpec], None, None]:
  logical_sizes = {
      name: shape[ax]
      for shape, in_axes in [(lhs_shape, spec.lhs_in_axes),
                             (rhs_shape, spec.rhs_in_axes)]
      for ax, name in in_axes.items()}
  yield from schedules(logical_sizes)


class PDotTests(jtu.JaxTestCase):

  def setUp(self):
    if not config.omnistaging_enabled:
      raise SkipTest("xmap requires omnistaging")
    super().setUp()

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBasic(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    f_mapped = xmap(f,
                    in_axes=[{1: 'i'}, {0: 'i'}],
                    out_axes={},
                    axis_resources={'i': 'r1'})

    rng = np.random.RandomState(0)
    x = rng.randn(3, 8)
    y = rng.randn(8, 5)

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.dot(x, y))

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBatching(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=[{0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}],
                    out_axes=['j', ...],
                    axis_resources={'i': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBatchingShardUncontractedDim(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=[{0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}],
                    out_axes=['j', ...],
                    axis_resources={'j': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{next(test_counter)}",
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "pdot_spec": pdot_spec,
       "axis_resources": axis_resources, "mesh_data": mesh_data}
      for test_counter in [it.count()]
      for lhs_shape, rhs_shape in product(
          [(2,), (2, 4, 2, 1)],
          repeat=2)
      for pdot_spec in all_pdot_specs(lhs_shape, rhs_shape)
      for axis_resources, mesh_data in schedules_from_pdot_spec(
          pdot_spec, lhs_shape, rhs_shape)))
  @ignore_xmap_warning()
  def testPdotSystematic(self, lhs_shape, rhs_shape, pdot_spec, axis_resources,
                         mesh_data):
    rng = jtu.rand_default(self.rng())
    lhs = rng(lhs_shape, np.float32)
    rhs = rng(rhs_shape, np.float32)

    def pdot_fun(x, y):
      # print(f'pdot(x:{x.aval.str_short()}, y:{y.aval.str_short()},\n'
      #       f'     axis_name={contract_names},\n'
      #       f'     pos_contract={spec.pos_contract_after_mapping}\n'
      #       f'     pos_batch={spec.pos_batch_after_mapping})')
      return jax.lax.pdot(x, y, axis_name=pdot_spec.contract_names,
                          pos_batch=pdot_spec.pos_batch_after_mapping,
                          pos_contract=pdot_spec.pos_contract_after_mapping)

    fun = xmap(pdot_fun, in_axes=[pdot_spec.lhs_in_axes, pdot_spec.rhs_in_axes],
               out_axes=[*pdot_spec.batch_names, ...],
               axis_resources=axis_resources)

    with with_mesh(mesh_data):
      result = fun(lhs, rhs)

    expected = lax.dot_general(lhs, rhs, pdot_spec.dot_general_dim_nums)
    tol = 1e-1 if jtu.device_under_test() == "tpu" else None
    self.assertAllClose(result, expected, check_dtypes=False,
                        atol=tol, rtol=tol)
  @ignore_xmap_warning()
  def test_xeinsum_vector_dot(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3)
    y = rng.randn(3)
    out = xmap(partial(jnp.einsum, '{i},{i}->'),
               in_axes=(['i', ...], ['i', ...]), out_axes=[...])(x, y)
    expected = np.einsum('i,i->', x, y)
    self.assertAllClose(out, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def test_xeinsum_outer_product(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3)
    y = rng.randn(3)
    out = xmap(partial(jnp.einsum, '{i},{j}->{i,j}'),
               in_axes=(['i', ...], ['j', ...]), out_axes=['i', 'j', ...])(x, y)
    expected = np.einsum('i,j->ij', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  @ignore_xmap_warning()
  def test_xeinsum_matmul(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 4)
    y = rng.randn(4, 5)

    out = xmap(partial(jnp.einsum, '{i,j},{j,k}->{i,k}'),
               in_axes=(['i', 'j', ...], ['j', 'k', ...]),
               out_axes=['i', 'k', ...])(x, y)
    expected = np.einsum('ij,jk->ik', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

    # order of named axes in the spec doesn't matter!
    out = xmap(partial(jnp.einsum, '{i,j},{k,j}->{k,i}'),
               in_axes=(['i', 'j', ...], ['j', 'k', ...]),
               out_axes=['i', 'k', ...])(x, y)
    expected = np.einsum('ij,jk->ik', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  def test_xeinsum_no_named_axes_vector_dot(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3)
    y = rng.randn(3)
    out = jnp.einsum('i,i->', x, y, _use_xeinsum=True)
    expected = np.einsum('i,i->', x, y)
    self.assertAllClose(out, expected, check_dtypes=False)

  def test_xeinsum_no_named_axes_batch_vector_dot(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 2)
    y = rng.randn(3, 2)
    out = jnp.einsum('ij,ij->i', x, y, _use_xeinsum=True)
    expected = np.einsum('ij,ij->i', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)

  def test_xeinsum_no_named_axes_reduce_sum(self):
    rng = np.random.RandomState(0)
    x = rng.randn(3)
    y = rng.randn()
    out = jnp.einsum('i,->', x, y, _use_xeinsum=True)
    expected = np.einsum('i,->', x, y)
    self.assertAllClose(out, expected, check_dtypes=True)


class XMapErrorTest(jtu.JaxTestCase):

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testRepeatedAxisResource(self):
    def f(v):
      return v * 4
    with self.assertRaisesRegex(ValueError, r"distinct resources.*specified \('x', 'x'\) for axis a"):
      fxy = xmap(f, in_axes=['a', ...], out_axes=['a', ...],
                 axis_resources={'a': ('x', 'x')})


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
