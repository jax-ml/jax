# Copyright 2020 The JAX Authors.
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
"""Microbenchmarks for JAX `api` functions."""

import enum
import functools
import operator

import google_benchmark
import jax
from jax import lax
from jax._src import config as jax_config
from jax.experimental import sparse
from jax._src.api_util import shaped_abstractify  # technically not an api fn
from jax._src.ad_checkpoint import checkpoint  # new jax.remat implementation
from jax._src.lib import xla_client as xc
from jax.interpreters import xla
from jax.interpreters import pxla
from jax._src import array
from jax._src import sharding
from jax.experimental import pjit as pjit_lib
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from jax.config import config

config.parse_flags_with_absl()


partial = functools.partial

def required_devices(num_devices_required):
  """Helper to skip benchmarks that require more devices."""
  def helper1(f):
    @functools.wraps(f)
    def helper2(state):
      if jax.device_count() < num_devices_required:
        state.skip_with_error(f"requires {num_devices_required} devices")
        return
      return f(state)
    return helper2
  return helper1


def create_mesh(shape, axis_names, state):
  size = np.prod(shape)
  if len(jax.devices()) < size:
    state.skip_with_error(f"Requires {size} devices")
    return None
  devices = sorted(jax.devices(), key=lambda d: d.id)
  mesh_devices = np.array(devices[:size]).reshape(shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


def swap(a, b):
  return b, a


class AnEnum(enum.IntEnum):
  A = 123
  B = 456

@google_benchmark.register
def eager_unary_dispatch(state):
  a = jax.device_put(1)
  lax.neg(a)
  while state:
    lax.neg(a)


@google_benchmark.register
def eager_unary(state):
  a = jax.device_put(1)
  lax.neg(a).block_until_ready()
  while state:
    lax.neg(a).block_until_ready()


@google_benchmark.register
def eager_binary_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b)
  while state:
    lax.add(a, b)


@google_benchmark.register
def eager_binary(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b).block_until_ready()
  while state:
    lax.add(a, b).block_until_ready()


@google_benchmark.register
def jit_trivial_dispatch(state):
  """Benchmarks only the duration for jitted_f to return the future."""
  f = jax.jit(swap)
  a, b = f(1, 2)
  x = f(a, b)
  while state:
    x = f(a, b)
  x[0].block_until_ready()


@google_benchmark.register
def jit_trivial(state):
  f = jax.jit(swap)
  a, b = f(1, 2)
  f(a, b)

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
def jit_simple_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b)


@google_benchmark.register
def jit_simple(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b).block_until_ready()

@google_benchmark.register
def jit_simple_dispatch_array(state):
  with jax_config.jax_array(True):
    a = jax.device_put(1)
    b = jax.device_put(2)
    f = jax.jit(operator.add)
    f(a, b)

    while state:
      f(a, b)


@google_benchmark.register
def jit_simple_array(state):
  with jax_config.jax_array(True):
    a = jax.device_put(1)
    b = jax.device_put(2)
    f = jax.jit(operator.add)
    f(a, b)

    while state:
      f(a, b).block_until_ready()


@google_benchmark.register
def jit_small_matmul(state):
  x = np.random.uniform(size=(2, 2)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


@google_benchmark.register
def jit_big_matmul(state):
  x = np.random.uniform(size=(100, 100)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'jax_array'])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@google_benchmark.option.args([1000, False])
@google_benchmark.option.args([1000, True])
@google_benchmark.option.args([2000, False])
@google_benchmark.option.args([2000, True])
def jit_simple_many_args_dispatch(state):
  with jax_config.jax_array(state.range(1)):
    args = [jax.device_put(i) for i in range(state.range(0))]
    f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
    x = f(args)
    x.block_until_ready()

    while state:
      x = f(args)
    x.block_until_ready()

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'jax_array'])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@google_benchmark.option.args([1000, False])
@google_benchmark.option.args([1000, True])
@google_benchmark.option.args([2000, False])
@google_benchmark.option.args([2000, True])
def jit_simple_many_args(state):
  with jax_config.jax_array(state.range(1)):
    args = [jax.device_put(i) for i in range(state.range(0))]
    f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
    f(args).block_until_ready()

    while state:
      f(args).block_until_ready()

def jit_simple_pruned_args_dispatch(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    x = f(*args)
  x.block_until_ready()


def jit_simple_pruned_args(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    f(*args).block_until_ready()

benchmarks = []
for n in [10, 100, 1000, 2000]:
  benchmarks += [
      google_benchmark.register(partial(jit_simple_pruned_args_dispatch, n),
                                name=f"jit_simple_pruned_args_dispatch_{n}"),
      google_benchmark.register(partial(jit_simple_pruned_args, n),
                                name=f"jit_simple_pruned_args_{n}")
  ]


@google_benchmark.register
def jit_dispatch_without_transfer(state):
  # We pick up a realistic input. 224 is usual for classification and 128 a
  # TPU-friendly batch-size.
  imgs = np.ones((128, 224, 224), np.float32)
  imgs = jax.device_put(imgs)

  f = jax.jit(lambda x: x+1)
  f(imgs)

  while state:
    f(imgs)


@google_benchmark.register
def jit_dispatch_with_transfer(state):
  imgs = np.ones((128, 224, 224), np.float32)

  f = jax.jit(lambda x: x+1)
  f(imgs).block_until_ready()

  while state:
    x = f(imgs)
  x.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(2)
def pmap_trivial_2_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(swap)
    a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

    while state:
      c, d = f(a, b)
      c.block_until_ready()
      d.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_trivial_dispatch_8_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(swap)
    a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
             jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

    while state:
      a, b = f(a, b)


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_trivial_8_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(swap)
    a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
             jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

    while state:
      c, d = f(a, b)
      c.block_until_ready()
      d.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(2)
def pmap_simple_2_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(lambda a, b: (a + b, a - b))
    a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

    while state:
      c, d = f(a, b)
      c.block_until_ready()
      d.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_simple_dispatch_8_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(lambda a, b: (a + b, a - b))
    a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
             jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

    while state:
      a, b = f(a, b)


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_simple_8_devices(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(lambda a, b: (a + b, a - b))
    a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
             jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

    while state:
      c, d = f(a, b)
      c.block_until_ready()
      d.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_simple_dispatch_8_devices_100_args(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
    args = []
    for i in range(100):
      args.append(jnp.array(list(range(i, i+8))))

    args = f(*args)

    while state:
      args = f(*args)


@google_benchmark.register
@google_benchmark.option.arg_name('jax_array')
@google_benchmark.option.arg(True)
@google_benchmark.option.arg(False)
@required_devices(8)
def pmap_simple_8_devices_100_args(state):
  with jax_config.jax_array(state.range(0)):
    f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
    args = []
    for i in range(100):
      args.append(jnp.array(list(range(i, i+8))))

    # Warmup loop.
    out = f(*args)

    while state:
      out = f(*args)
      jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)


def _run_sda_index_bench(state, num_devices):
  x = jax.pmap(jnp.sin)(jnp.arange(num_devices))
  jax.device_get(x)
  while state:
    for i in range(num_devices):
      _ = x[i]


@google_benchmark.register
@required_devices(1)
def sda_index_1(state):
  _run_sda_index_bench(state, 1)


@google_benchmark.register
@required_devices(2)
def sda_index_2(state):
  _run_sda_index_bench(state, 2)


@google_benchmark.register
@required_devices(8)
def sda_index_8(state):
  _run_sda_index_bench(state, 8)


def _sparse_bcoo_fromdense(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  size = np.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape)
  mat = jnp.zeros(shape).at[indices].set(data)

  f = sparse.BCOO.fromdense
  if compile or jit:
    # Note: nse must be specified for JIT.
    f = jax.jit(partial(f, nse=nse))

  if compile:
    while state:
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


@google_benchmark.register
def sparse_bcoo_fromdense(state):
  return _sparse_bcoo_fromdense(state)


@google_benchmark.register
def sparse_bcoo_fromdense_jit(state):
  return _sparse_bcoo_fromdense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_fromdense_compile(state):
  return _sparse_bcoo_fromdense(state, compile=True)


def _sparse_bcoo_todense(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  size = np.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape)
  mat = sparse.BCOO((jnp.array(data), jnp.column_stack(indices)), shape=shape)

  f = lambda mat: mat.todense()
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


@google_benchmark.register
def sparse_bcoo_todense(state):
  return _sparse_bcoo_todense(state)


@google_benchmark.register
def sparse_bcoo_todense_jit(state):
  return _sparse_bcoo_todense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_todense_compile(state):
  return _sparse_bcoo_todense(state, compile=True)


def _sparse_bcoo_matvec(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  key = jax.random.PRNGKey(1701)
  mat = sparse.random_bcoo(key, nse=nse, shape=shape, dtype=jnp.float32,
                           indices_dtype=jnp.int32, sorted_indices=True)
  vec = jax.random.uniform(key, shape=(shape[1],), dtype=jnp.float32)

  f = lambda mat, vec: mat @ vec
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      f.lower(mat, vec).compile()
  else:
    f(mat, vec).block_until_ready()
    while state:
      f(mat, vec).block_until_ready()


@google_benchmark.register
def sparse_bcoo_matvec(state):
  return _sparse_bcoo_matvec(state)


@google_benchmark.register
def sparse_bcoo_matvec_jit(state):
  return _sparse_bcoo_matvec(state, jit=True)


@google_benchmark.register
def sparse_bcoo_matvec_compile(state):
  return _sparse_bcoo_matvec(state, compile=True)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_shaped_abstractify(state):
  device, *_ = jax.devices()
  args = [jax.device_put_replicated(1, [device])] * 1000
  while state:
    _ = [shaped_abstractify(x) for x in args]


def _run_benchmark_for_xla_abstractify(arg, state):
  while state:
    xla.abstractify(arg)

def bench_xla_abstractify():
  _abstractify_args = [
      (3, 'scalar_int'),
      (3.5, 'scalar_float'),
      (np.int32(3), 'scalar_numpy_int32'),
      (np.uint32(7), 'scalar_numpy_uint32'),
      (np.random.randn(3, 4, 5, 6), 'numpy_random'),
      (np.arange(100, dtype=np.float32), 'numpy_arange_100_float32'),
      (AnEnum.B, 'enum'),
  ]
  benchmarks = []
  for a, name in _abstractify_args:
    benchmarks.extend([
        google_benchmark.register(
            partial(_run_benchmark_for_xla_abstractify, a),
            name=f'bench_xla_abstractify_{name}'),
    ])
bench_xla_abstractify()


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
def bench_are_op_shardings_equal(state):
  op1 = xc.OpSharding()
  op1.type = xc.OpSharding.Type.OTHER
  op1.tile_assignment_dimensions = [4, 192, 16]
  op1.tile_assignment_devices = list(range(12288))

  op2 = xc.OpSharding()
  op2.type = xc.OpSharding.Type.OTHER
  op2.tile_assignment_dimensions = [4, 192, 16]
  op2.tile_assignment_devices = list(range(12288))

  while state:
    pxla.are_op_shardings_equal(op1, op2)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_pjit_check_aval_sharding(state):
  mesh = create_mesh((4, 2), ('x', 'y'), state)
  if mesh is None:
    return
  s = sharding.NamedSharding(mesh, pxla.PartitionSpec('x', 'y'))
  aval = jax.core.ShapedArray((8, 2), np.int32)

  while state:
    pjit_lib.pjit_check_aval_sharding([s] * 100, [aval] * 100, 'benchmark', False)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_remat_eager_retracing_overheads(state):
  def double_compose(f):
    return lambda x: f(f(x))

  f = jnp.sin
  for _ in range(6):
    f = double_compose(f)
  f = double_compose(checkpoint(f))

  while state:
    y, _ = jax.vjp(f, 3.)
  y.block_until_ready()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_remat_eager_retracing_overheads_static_argnums(state):
  def double_compose(f):
    return lambda x, y: f(f(x, y), y)

  f = lambda x, _: jnp.sin(x)
  for _ in range(6):
    f = double_compose(f)
  f = double_compose(checkpoint(f, static_argnums=(1,)))

  while state:
    y, _ = jax.vjp(f, 3., True)
  y.block_until_ready()


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_slicing_compilation(state):
  x = jnp.arange(3)
  while state:
    jax.jit(lambda x: (x[0], x[1], x[2])).lower(x).compile()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_slicing_compilation2(state):
  x = jnp.arange(3)
  while state:
    jax.jit(lambda x: (x[:1], x[1:2], x[2:3])).lower(x).compile()

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_repeated_static_indexing(state):
  x = jnp.arange(500)
  while state:
    jax.block_until_ready([x[i] for i in range(500)])

@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def bench_repeated_static_slicing(state):
  x = jnp.arange(1000)
  while state:
    jax.block_until_ready([x[i:i + 2] for i in range(0, 1000, 2)])

def pjit_simple_benchmark(state, num_devices, num_args, cpp_jit, use_aot=False):
  spec = jax.sharding.PartitionSpec('x')
  mesh = create_mesh((num_devices,), ('x',), state)
  if mesh is None:
    return
  s = sharding.NamedSharding(mesh, spec)
  inp_data = np.arange(num_devices).astype(np.float32)
  x = array.make_array_from_callback(inp_data.shape, s, lambda idx: inp_data[idx])

  x = [x for _ in range(num_args)]

  prev_state = jax_config.FLAGS.experimental_cpp_pjit
  jax_config.FLAGS.experimental_cpp_pjit = cpp_jit

  in_axis_resources = sharding.NamedSharding(mesh, spec)
  out_axis_resources = sharding.NamedSharding(mesh, spec)

  f = pjit_lib.pjit(
      lambda x: jax.tree_map(lambda x: x + 1, x),
      in_axis_resources=in_axis_resources,
      out_axis_resources=out_axis_resources)

  if use_aot:
    f = f.lower(x).compile()

  x = f(x)

  while state:
    x = f(x)

  jax_config.FLAGS.experimental_cpp_pjit = prev_state


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_simple_1_device(state):
  pjit_simple_benchmark(
      state, num_devices=1, num_args=state.range(0), cpp_jit=state.range(1))

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_simple_4_device(state):
  pjit_simple_benchmark(
      state, num_devices=4, num_args=state.range(0), cpp_jit=state.range(1))

@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_simple_4000_device(state):
  pjit_simple_benchmark(
      state, num_devices=4000, num_args=state.range(0), cpp_jit=state.range(1))


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_aot_1_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=1,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_aot_4_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=4,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.arg_names(['num_args', 'cpp_pjit'])
@google_benchmark.option.args([1, False])
@google_benchmark.option.args([1, True])
@google_benchmark.option.args([10, False])
@google_benchmark.option.args([10, True])
@google_benchmark.option.args([100, False])
@google_benchmark.option.args([100, True])
@jax_config.jax_array(True)
def pjit_aot_4000_device(state):
  pjit_simple_benchmark(
      state,
      num_devices=4000,
      num_args=state.range(0),
      cpp_jit=state.range(1),
      use_aot=True)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def host_local_array_to_global_array(state):
  global_mesh = create_mesh((4, 2), ('x', 'y'), state)
  input_shape = (8, 2)
  input_data = np.arange(np.prod(input_shape)).reshape(input_shape)
  in_pspec = pxla.PartitionSpec('x', 'y')

  while state:
    multihost_utils.host_local_array_to_global_array(
        (input_data, input_data), global_mesh, (in_pspec, in_pspec))

@google_benchmark.register
def device_put(state):
  x = np.array(1, np.int32)
  while state:
    _ = jax.device_put(x).block_until_ready()


def batch_inplace_while(inplace_op, state):

  @jax.jit
  @jax.vmap
  def f(init_step, init_xs):

    def cond(carry):
      step, xs = carry
      return step < xs.size

    def body(carry):
      step, xs = carry
      if inplace_op == 'scatter':
        xs = xs.at[step].set(1)
      elif inplace_op == 'dynamic_update_slice':
        xs = lax.dynamic_update_index_in_dim(xs, 1., step, 0)
      else:
        assert False
      return step + 1, xs

    return lax.while_loop(cond, body, (init_step, init_xs))

  size = 100_000
  args = jnp.array([0]), jnp.zeros((1, size))
  f(*args)  # compile
  while state:
    f(*args)


google_benchmark.register(
    partial(batch_inplace_while, 'scatter'), name='batch_inplace_while_scatter')
google_benchmark.register(
    partial(batch_inplace_while, 'dynamic_update_slice'),
    name='batch_inplace_while_dynamic_update_slice')

if __name__ == "__main__":
  google_benchmark.main()
