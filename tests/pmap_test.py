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

import jax
import jax.numpy as np
from jax import test_util as jtu
from jax import core
from jax import lax
from jax import random
from jax.api import (pmap, soft_pmap, jit, vmap, jvp, grad, make_jaxpr,
                     linearize, device_put)
from jax.lib import xla_bridge
from jax.util import prod
from jax.interpreters import pxla
from jax.interpreters import xla

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

  def testComplexPsum(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4 * 2)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape).view(onp.complex64)
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

  def testMismatchedAxisSizes(self):
    n = xla_bridge.device_count()
    f = pmap(lambda x, y: x + y)
    self.assertRaisesRegexp(
        ValueError,
        "Axis size .* does not match leading dimension of shape .*",
        lambda: f(onp.random.randn(n), onp.random.randn(n - 1)))

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

  def testGradOfPsum(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return lax.psum(x, axis_name='i')

    shape = (jax.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    jtu.check_grads(f, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, eps=1.)

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

  def testAxisGroups(self):
    axis_env = xla.AxisEnv(8, ['i', 'j'], [4, 2])
    groups = xla.axis_groups(axis_env, 'i')
    self.assertEqual(groups, ((0, 2, 4, 6), (1, 3, 5, 7)))

    groups = xla.axis_groups(axis_env, 'j')
    self.assertEqual(groups, ((0, 1), (2, 3), (4, 5), (6, 7)))

    groups = xla.axis_groups(axis_env, ('i', 'j'))
    self.assertEqual(groups, ((0, 1, 2, 3, 4, 5, 6, 7,),))

    groups = xla.axis_groups(axis_env, ('j', 'i'))
    self.assertEqual(len(groups), 1)
    self.assertEqual((tuple(sorted(groups[0])),),
                     ((0, 1, 2, 3, 4, 5, 6, 7,),))  # order doesn't matter

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
  def testIssue1703(self):
    num_devices = xla_bridge.device_count()
    perm = [num_devices - 1] + list(range(num_devices - 1))
    f = pmap(
      lambda x: lax.ppermute(x, "i", zip(range(num_devices), perm)), "i")
    result = f(np.arange(num_devices))
    expected = np.asarray(perm, dtype=np.float32)
    self.assertAllClose(result, expected)

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
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = pmap(lambda x: x)
    x = onp.ones((device_count + 1, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = pmap(lambda x: pmap(lambda x: x)(x))
    x = onp.ones((device_count, 2, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

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

  def testVmapOfPmap2(self):
    N_DEVICES = xla_bridge.device_count()
    keys = random.split(random.PRNGKey(1), 13)  # [13, 2]

    @pmap
    def g(key):
      params = random.normal(key, ())
      return 0.

    @vmap
    def s(keys):
      keys = np.broadcast_to(keys, (N_DEVICES,) + keys.shape)
      return g(keys)

    ans = s(keys)  # doesn't crash
    self.assertEqual(ans.shape, (13, N_DEVICES))

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

  def testSoftPmapDevicePersistence(self):
    device_count = xla_bridge.device_count()
    shape = (2 * 2 * device_count, 2, 3)

    # check that we can maintain device persistence across calls
    x = onp.arange(prod(shape)).reshape(shape)
    x = soft_pmap(lambda x: x)(x)
    self.assertIsInstance(x, pxla.ChunkedDeviceArray)
    x._npy_value = onp.float32(onp.nan)  # can't be coerced to ndarray for xfer
    x = soft_pmap(lambda x: x)(x)  # doesn't crash
    self.assertIsInstance(x, pxla.ChunkedDeviceArray)

    # check that we don't crash when we can't maintain device persistence
    x = onp.arange(prod(shape)).reshape(shape)
    x = soft_pmap(lambda x: x)(x)
    self.assertIsInstance(x, pxla.ChunkedDeviceArray)
    y = x.reshape(device_count, -1)
    self.assertIsInstance(y, xla.DeviceArray)  # should have forced collection
    soft_pmap(lambda x: x)(y)  # doesn't crash
    z = x + 2
    self.assertIsInstance(z, xla.DeviceArray)  # should have forced collection
    x._npy_value = onp.float32(onp.nan)  # can't be coerced to ndarray for xfer
    self.assertRaisesRegex(
        RuntimeError,
        '.*does not match host shape or layout of computation parameter 0.*',
        lambda: x + 2)

    # check that different axis merges aren't a problem
    x = onp.arange(prod(shape)).reshape(shape)
    x = soft_pmap(lambda x: x)(x)
    self.assertIsInstance(x, pxla.ChunkedDeviceArray)
    x = x.reshape(2 * device_count, 2, 2, 3)  # axis merge of the wrong size
    self.assertIsInstance(x, xla.DeviceArray)  # should have forced collection

  @jtu.skip_on_devices("gpu")
  def DISABLED_testSoftPmapAllToAll(self):
    n = 4 * xla_bridge.device_count()
    def f(x):
      return lax.all_to_all(x, 'i', 0, 0)
    ans = soft_pmap(f, 'i')(np.arange(n ** 2).reshape(n, n))
    expected = onp.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testShardedDeviceArrayBlockUntilReady(self):
    x = onp.arange(xla_bridge.device_count())
    x = pmap(lambda x: x)(x)
    x.block_until_ready()  # doesn't crash

  def testJitPmapComposition(self):
    f = lambda x: x - lax.psum(x, 'i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.sum(x, 0)

    ans = jit(pmap(f, 'i'))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = pmap(jit(f), 'i')(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMakeJaxprOfOpenSpmd(self):
    f = lambda x: x - lax.psum(x, 'i')
    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    make_jaxpr(f)(x)  # doesn't crash

  def testCompositionWithJitTwice(self):
    @jit
    def f(x):
      y = 2 * x

      @jit
      def g(z):
        return pmap(lambda x: x * y)(z)

      return g(x)

    f(onp.arange(1.).reshape((1, 1)))  # doesn't crash

  def testIssue1065(self):
    # from https://github.com/google/jax/issues/1065
    device_count = xla_bridge.device_count()

    def multi_step_pmap(state, count):
      @partial(pmap, axis_name='x')
      @jit
      def exchange_and_multi_step(state):
        return state

      @jit
      def time_evolution(state):
        return lax.fori_loop(0, count, lambda i, s: exchange_and_multi_step(s), state)

      return time_evolution(state)

    multi_step_pmap(np.zeros((device_count,)), count=1)

  def testShardedDeviceArrayGetItem(self):
    f = lambda x: 2 * x
    f = pmap(f, axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    y = f(x)
    self.assertIsInstance(y, np.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)

    z = y[0]  # doesn't crash
    self.assertAllClose(z, 2 * x[0], check_dtypes=False)

  def testPostProcessMap(self):
    # TODO(mattjj): this fails with multiple devices (unless we add a jit)
    # because we assume eager ops (like scan here) can't require more than 1
    # replica.
    raise SkipTest("need eager multi-replica support")
    # test came from https://github.com/google/jax/issues/1369
    nrep = xla_bridge.device_count()

    def pmvm(a, b):
      a = a.reshape((nrep, -1, a.shape[1]))
      func = pmap(lambda z: np.dot(z, b))
      return func(a).reshape(b.shape)

    n = nrep * 2
    rng = onp.random.RandomState(0)
    a = rng.randn(n, n)
    b = rng.randn(n)

    iters = np.arange(5)
    def body(carry, i):
      return pmvm(a, carry), i
    ans, _ = lax.scan(body, b, iters)

    expected = onp.linalg.matrix_power(a, 5).dot(b)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testManyArgs(self):
    @pmap
    def f(args_list):
      return sum(args_list)

    vals = list(range(500))
    ndevices = xla_bridge.device_count()
    self.assertAllClose(f(np.array([vals] * ndevices)),
                        np.array([sum(vals)] * ndevices),
                        check_dtypes=True)


class PmapWithDevicesTest(jtu.JaxTestCase):

  def testAllDevices(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i',
             devices=xla_bridge.devices())
    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.sum(x, 0)
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testOneDevice(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    d0 = xla_bridge.devices()[0]
    d1 = xla_bridge.devices()[1]
    f = lambda x: np.dot(x, x.T)
    f0 = pmap(f, devices=[d0])
    f1 = pmap(f, devices=[d1])
    x = onp.random.rand(1, 1000, 1000)
    r0 = f0(x)
    r1 = f1(x)
    expected = onp.expand_dims(onp.dot(x.squeeze(), x.squeeze().T), 0)
    self.assertAllClose(r0, expected, check_dtypes=True)
    self.assertAllClose(r1, expected, check_dtypes=True)

  def testNoDevicesError(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i', devices=[])
    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    with self.assertRaisesRegex(
        ValueError, "'devices' argument to pmap must be non-empty, or None."):
      f(x)

  def testBadAxisSizeError(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    f = pmap(lambda x: lax.psum(x, 'i'), axis_name='i',
             devices=xla_bridge.devices())
    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=1, "
        r"num_local_devices=\d."):
      f(np.ones(1))

    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=\d, "
        r"num_local_devices=\d."):
      f(np.ones(xla_bridge.device_count() + 1))

  def testNestedPmapsError(self):
    # Devices specified in outer pmap
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def foo(x):
      @partial(pmap, axis_name='j')
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    with self.assertRaisesRegex(
        ValueError,
        "Nested pmaps with explicit devices argument."):
      foo(np.ones((xla_bridge.device_count(), 1)))

    # Devices specified in inner pmap
    @partial(pmap, axis_name='i')
    def foo(x):
      @partial(pmap, axis_name='j', devices=xla_bridge.devices())
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    with self.assertRaisesRegex(
        ValueError,
        "Nested pmaps with explicit devices argument."):
      foo(np.ones((xla_bridge.device_count(), 1)))

  def testJitInPmap(self):
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def foo(x):
      @jit
      def bar(y):
        return y + 1
      return lax.psum(bar(x), 'i')

    ndevices = xla_bridge.device_count()
    ans = foo(np.ones((ndevices, 1)))
    expected = onp.ones((ndevices, 1)) * ndevices * 2
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testPmapInJit(self):
    @jit
    def foo(x):
      @partial(pmap, axis_name='i', devices=xla_bridge.devices())
      def bar(y):
        return lax.psum(y, 'i')
      return bar(x)

    ndevices = xla_bridge.device_count()
    ans = foo(np.ones((ndevices, 1)))
    expected = onp.ones((ndevices, 1)) * ndevices
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testGradBasic(self):
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def f(x):
      return np.sin(x)

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(np.sin(x)))(x)
    expected = grad(lambda x: np.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
