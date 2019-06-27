# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from unittest import SkipTest

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import core
from jax import lax
from jax.api import (pmap, soft_pmap, jit, vmap, jvp, grad, make_jaxpr,
                     linearize, device_put)
from jax.lib import xla_bridge
from jax.util import prod
from jax.interpreters import pxla

from jax.config import config
config.parse_flags_with_absl()


class PmapTest(jtu.JaxTestCase):

  def _getMeshShape(self, device_mesh_shape):
    device_count = xla_bridge.device_count()
    if any(size == -1 for size in device_mesh_shape):
      try:
        return onp.arange(device_count).reshape(device_mesh_shape).shape
      except ValueError:
        msg = "device mesh shape {} not compatible with device count {}"
        raise SkipTest(msg.format(device_mesh_shape, device_count))
    else:
      if device_count % prod(device_mesh_shape):
        msg = "device mesh size {} does not divide available device count {}"
        raise SkipTest(msg.format(prod(device_mesh_shape), device_count))
      else:
        return device_mesh_shape

  def testBasic(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNestedBasic(self):
    f = lambda x: lax.psum(lax.psum(x, 'i'), 'j')
    f = pmap(pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return onp.repeat(onp.sum(x, axis, keepdims=True), x.shape[axis], axis)

    shape = (xla_bridge.device_count(), 1, 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedShardingAndStacking(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = lambda x: x
    f = pmap(pmap(f, 'i'), 'j')

    shape = mesh_shape + (4,)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = f(x)
    expected = x
    self.assertEqual(ans.shape, expected.shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJvpAndPartialEval(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(np.ones_like(x))

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = onp.cos(x)

    ans = splitjvp(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    make_jaxpr(splitjvp)(x)  # doesn't crash

  def testGradBasic(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(np.sin(x)))(x)
    expected = grad(lambda x: np.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfJvp(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(np.ones_like(x))

    fun = lambda x: np.sum(jvp(np.sin, (x,), (np.ones_like(x),))[1])

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(splitjvp(x)))(x)
    expected = grad(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testTwoArgsGrad(self):
    def f(x, y):
      return lax.psum(5. * np.cos(x) * np.sin(y), 'i')
    f = pmap(f, 'i')

    def g(x, y):
      tot = np.sum(5. * np.cos(x) * np.sin(y))
      return tot * np.ones_like(x)  # broadcast to map like pjit does

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    y = 4 + x
    ans = grad(lambda x, y: np.sum(g(x, y)))(x, y)
    expected = grad(lambda x, y: np.sum(g(x, y)))(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedWithClosure(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    @partial(pmap, axis_name='i')
    def test_fun(x):
      y = np.sum(np.sin(x))

      @partial(pmap, axis_name='j')
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    @vmap
    def baseline_fun(x):
      y = np.sum(np.sin(x))

      @vmap
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    shape = mesh_shape + (4,)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(test_fun(x)))(x)
    expected = grad(lambda x: np.sum(baseline_fun(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testShardedDeviceArrays(self):
    f = lambda x: 2 * x
    f = pmap(f, axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    # test that we can pass in and out ShardedDeviceArrays
    y = f(x)
    self.assertIsInstance(y, np.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)
    z = f(y)
    self.assertIsInstance(z, pxla.ShardedDeviceArray)
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can pass in a regular DeviceArray
    y = f(device_put(x))
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

    # test that we can pass a ShardedDeviceArray to a regular jit computation
    z = y + y
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can handle device movement on dispatch
    y.device_buffers = y.device_buffers[::-1]
    z = f(y)
    self.assertAllClose(z, 2 * 2 * x[::-1], check_dtypes=False)

    # test that the repr doesn't crash
    repr(z)

  def testPsumMultiple(self):
    f = lambda x: lax.psum(x, ('i', 'j'))
    f = pmap(pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return onp.repeat(onp.sum(x, axis, keepdims=True), x.shape[axis], axis)

    device_count = xla_bridge.device_count()
    num_pairs, ragged = divmod(device_count, 2)
    if num_pairs > 1 and not ragged:
      shape = (num_pairs, 2, 4)
    else:
      shape = (device_count, 1, 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReplicaGroups(self):
    groups = pxla.replica_groups(8, [4, 2], (0,))
    self.assertEqual(groups, ((0, 2, 4, 6), (1, 3, 5, 7)))

    groups = pxla.replica_groups(8, [4, 2], (1,))
    self.assertEqual(groups, ((0, 1), (2, 3), (4, 5), (6, 7)))

    groups = pxla.replica_groups(8, [4, 2], (0, 1))
    self.assertEqual(groups, ((0, 1, 2, 3, 4, 5, 6, 7,),))

    groups = pxla.replica_groups(8, [4, 2], (1, 0))
    self.assertEqual(len(groups), 1)
    self.assertEqual((tuple(sorted(groups[0])),),
                     ((0, 1, 2, 3, 4, 5, 6, 7,),))  # order doesn't matter

  def testShardedDeviceTuple(self):
    f = lambda x: core.pack((x, x))
    f = pmap(f)

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    # test that we can pass in and out ShardedDeviceTuples (and unpack them)
    y = f(x)
    self.assertIsInstance(y, pxla.ShardedDeviceTuple)
    self.assertIsInstance(y, core.JaxTuple)
    self.assertAllClose(y, (x, x), check_dtypes=False)
    z = f(y)
    self.assertIsInstance(z, pxla.ShardedDeviceTuple)
    self.assertAllClose(z, (y, y), check_dtypes=True)

    # test that we can pass a ShardedDeviceTuple to a regular jit computation
    w = jit(lambda x: list(x)[0])(y)
    self.assertAllClose(w, x, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermute(self):
    device_count = xla_bridge.device_count()
    rotation = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=rotation, axis_name='i')
    f = pmap(f, 'i')

    x = np.arange(4 * device_count).reshape((device_count, 4))
    ans = f(x)
    expected = onp.roll(x, shift=1, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermuteGrad(self):
    device_count = xla_bridge.device_count()
    shift_right = [(i, (i + 1)) for i in range(device_count - 1)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = onp.pi + onp.arange(device_count, dtype=onp.float32)
    g = lambda x: np.sum(y * pmap(f, 'i')(x))

    x = onp.arange(device_count, dtype=onp.float32)
    ans = grad(g)(x)
    expected = onp.concatenate([onp.pi + onp.arange(1, device_count), [0]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermuteCyclicGrad(self):
    device_count = xla_bridge.device_count()
    shift_right = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = onp.pi + onp.arange(device_count, dtype=onp.float32)
    g = lambda x: np.sum(y * pmap(f, 'i')(x))

    x = onp.arange(device_count, dtype=onp.float32)
    ans = grad(g)(x)
    expected = onp.roll(onp.pi + onp.arange(device_count), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testRule30(self):
    # This is a test of collective_permute implementing a simple halo exchange
    # to run a rule 30 simulation: https://en.wikipedia.org/wiki/Rule_30
    # Halo exchange should be useful in spatially-sharded convolutions and in
    # other simulations.
    device_count = xla_bridge.device_count()

    def send_right(x, axis_name):
      left_perm = [(i, (i + 1) % device_count) for i in range(device_count)]
      return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

    def send_left(x, axis_name):
      left_perm = [((i + 1) % device_count, i) for i in range(device_count)]
      return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

    def update_board(board):
      left = board[:-2]
      right = board[2:]
      center = board[1:-1]
      return lax.bitwise_xor(left, lax.bitwise_or(center, right))

    @partial(pmap, axis_name='i')
    def step(board_slice):
      left, right = board_slice[:1], board_slice[-1:]
      right, left = send_left(left, 'i'), send_right(right, 'i')
      enlarged_board_slice = np.concatenate([left, board_slice, right])
      return update_board(enlarged_board_slice)

    board = onp.zeros(40, dtype=bool)
    board[board.shape[0] // 2] = True
    reshaped_board = board.reshape((device_count, -1))

    boards = []
    def print_board(board):
      boards.append(''.join('*' if x else ' ' for x in board.ravel()))

    print_board(reshaped_board)
    for _ in range(20):
      reshaped_board = step(reshaped_board)
      print_board(reshaped_board)

    ans = '\n'.join(boards)
    expected = '\n'.join((
        '                    *                   ',
        '                   ***                  ',
        '                  **  *                 ',
        '                 ** ****                ',
        '                **  *   *               ',
        '               ** **** ***              ',
        '              **  *    *  *             ',
        '             ** ****  ******            ',
        '            **  *   ***     *           ',
        '           ** **** **  *   ***          ',
        '          **  *    * **** **  *         ',
        '         ** ****  ** *    * ****        ',
        '        **  *   ***  **  ** *   *       ',
        '       ** **** **  *** ***  ** ***      ',
        '      **  *    * ***   *  ***  *  *     ',
        '     ** ****  ** *  * *****  *******    ',
        '    **  *   ***  **** *    ***      *   ',
        '   ** **** **  ***    **  **  *    ***  ',
        '  **  *    * ***  *  ** *** ****  **  * ',
        ' ** ****  ** *  ******  *   *   *** ****',
        ' *  *   ***  ****     **** *** **   *   ',
    ))

    print(ans)
    self.assertEqual(ans, expected)

  @jtu.skip_on_devices("cpu", "gpu")
  def testReduceMax(self):
    f = pmap(lambda x: x - lax.pmax(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.max(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testReduceMin(self):
    f = pmap(lambda x: x - lax.pmin(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.min(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDeviceCountError(self):
    device_count = xla_bridge.device_count()

    f = pmap(lambda x: x)
    x = np.arange(device_count + 1)
    self.assertRaisesRegexp(
        ValueError,
        ".*requires.*replicas",
        lambda: f(x))

    f = pmap(lambda x: x)
    x = onp.ones((device_count + 1, 10))
    self.assertRaisesRegexp(
        ValueError,
        ".*requires.*replicas",
        lambda: f(x))

    f = pmap(lambda x: pmap(lambda x: x)(x))
    x = onp.ones((device_count, 2, 10))
    self.assertRaisesRegexp(
        ValueError,
        ".*requires.*replicas",
        lambda: f(x))

  def testPmapConstant(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: 3)
    x = np.arange(device_count)
    ans = f(x)
    expected = onp.repeat(3, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCollectiveConstant(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: lax.psum(1, 'i'), 'i')
    x = np.arange(device_count)
    ans = f(x)
    expected = onp.repeat(device_count, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCollectiveConstantNested(self):
    device_count = xla_bridge.device_count()

    @partial(pmap, axis_name='i')
    def f(x):
      @partial(pmap, axis_name='j')
      def g(y):
        a = lax.psum(1, 'i')
        b = lax.psum(1, 'j')
        c = lax.psum(1, ('i', 'j'))
        return a, b, c
      return g(x)

    shape = (device_count, 1, 4)
    x = np.arange(prod(shape)).reshape(shape)
    a, b, c = f(x)

    self.assertEqual(a.shape, shape[:-1])
    self.assertEqual(b.shape, shape[:-1])
    self.assertEqual(c.shape, shape[:-1])

    self.assertEqual(a.ravel()[0], device_count)
    self.assertEqual(b.ravel()[0], 1)
    self.assertEqual(c.ravel()[0], device_count * 1)

  def testAxisIndex(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: x + pxla.axis_index('i'), 'i')
    x = np.ones(device_count)
    ans = f(x)
    expected = 1 + onp.arange(device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testVmapOfPmap(self):
    device_count = xla_bridge.device_count()
    f0 = lambda x: x
    f1 = pmap(f0, axis_name='i')
    ax = onp.random.randn(2, device_count, 50, 60)
    bx = vmap(f1)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmapNonLeadingAxis(self):
    device_count = xla_bridge.device_count()
    f0 = lambda x: x
    f1 = pmap(f0, axis_name='i')
    ax = onp.random.randn(device_count, 2, 50, 60)
    bx = vmap(f1, in_axes=2, out_axes=2)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmapTuple(self):
    device_count = xla_bridge.device_count()
    f0 = lambda *x: x
    f1 = pmap(f0, axis_name='i')

    ax = onp.random.randn(device_count, 2, 50, 60)
    ay = onp.random.randn(device_count, 30, 2)
    az1 = onp.random.randn(device_count, 20)
    az2 = onp.random.randn(2, device_count, 20)

    bx, by, bz = vmap(f1, in_axes=(1, 2, (None, 0)), out_axes=(1, 2, 0))(ax, ay, (az1, az2))

    self.assertAllClose(ax, bx, check_dtypes=False)
    self.assertAllClose(ay, by, check_dtypes=False)

    bz1, bz2 = bz
    expected_bz1 = onp.broadcast_to(az1, (2,) + az1.shape)
    self.assertAllClose(expected_bz1, bz1, check_dtypes=False)
    self.assertAllClose(bz2, bz2, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testPswapaxes(self):
    device_count = xla_bridge.device_count()
    shape = (device_count, 3, device_count, 5)
    x = onp.arange(prod(shape)).reshape(shape)

    ans = pmap(lambda x: lax.pswapaxes(x, 'i', 1), axis_name='i')(x)
    expected = onp.swapaxes(x, 0, 2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSoftPmapPsum(self):
    n = 4 * xla_bridge.device_count()
    def f(x):
      return x / lax.psum(x, 'i')
    ans = soft_pmap(f, 'i')(np.ones(n))
    expected = onp.ones(n) / n
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSoftPmapAxisIndex(self):
    n = 4 * xla_bridge.device_count()
    def f(x):
      return x * lax.axis_index('i')
    ans = soft_pmap(f, 'i')(2 * np.ones(n))
    expected = 2 * onp.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSoftPmapOfJit(self):
    n = 4 * xla_bridge.device_count()
    def f(x):
      return 3 * x
    ans = soft_pmap(jit(f), 'i')(onp.arange(n))
    expected = 3 * onp.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSoftPmapNested(self):
    n = 4 * xla_bridge.device_count()

    @partial(soft_pmap, axis_name='i')
    @partial(soft_pmap, axis_name='j')
    def f(x):
      i_size = lax.psum(1, 'i')
      return x + lax.axis_index('i') + i_size * lax.axis_index('j')

    ans = f(np.zeros((n, n)))
    expected = onp.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfSoftPmap(self):
    n = 4 * xla_bridge.device_count()

    @partial(soft_pmap, axis_name='i')
    def f(x):
      return x * lax.axis_index('i')

    ans = grad(lambda x: np.sum(f(x)))(np.zeros((n, n)))
    expected = onp.repeat(onp.arange(n)[:, None], n, axis=1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testSoftPmapAllToAll(self):
    n = 4 * xla_bridge.device_count()
    def f(x):
      return lax.all_to_all(x, 'i', 0, 0)
    ans = soft_pmap(f, 'i')(np.arange(n ** 2).reshape(n, n))
    expected = onp.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
