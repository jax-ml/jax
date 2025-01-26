# Copyright 2019 The JAX Authors.
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

"""Tests for nn module."""

import collections
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import scipy.stats

from jax._src import ad_checkpoint
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
from jax.test_util import check_grads
from jax import nn
from jax import random
import jax
import jax.numpy as jnp

config.parse_flags_with_absl()

def _is_required_cudnn_version_satisfied(min_cudnn_version):
  return (
      jtu.is_cuda_compute_capability_at_least("8.0") and
      cuda_versions is not None and
      cuda_versions.cudnn_get_version() >= min_cudnn_version
  )

def _check_cudnn_backend(fn, *args, **kwargs):
  lowered = jax.jit(fn).lower(*args, **kwargs)
  hlo = lowered.as_text('stablehlo', debug_info=True)
  return '__cudnn$fmha' in hlo

_cudnn_dbias_error = 'cuDNN only supports bias gradient'

@jtu.with_config(jax_legacy_prng_key="allow",
                 jax_numpy_dtype_promotion="standard")
class NNFunctionsTest(jtu.JaxTestCase):
  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float16],
      group_num=[1, 2, 4],
      use_vmap=[False, True],
      impl=['cudnn', 'xla'],
  )
  def testDotProductAttention(self, dtype, group_num, use_vmap, impl):
    if impl == 'cudnn' and not _is_required_cudnn_version_satisfied(8904):
      raise unittest.SkipTest("CUDA or cuDNN versions are not compatible.")
    if impl == 'cudnn' and dtype == jnp.float32:
      raise unittest.SkipTest("cuDNN only supports fp16 or bf16.")

    B, S, T, N, H, G = 2, 128, 128, 4, 32, group_num
    keys = random.split(random.PRNGKey(0), 5)
    Q = random.normal(keys[0], (B, T, N, H), dtype)
    K = random.normal(keys[1], (B, S, N // G, H), dtype)
    V = random.normal(keys[2], (B, S, N // G, H), dtype)
    grad = random.normal(keys[3], (B, T, N, H), dtype)
    bias, mask = None, None

    sdpa = nn.dot_product_attention
    sdpa_ref = partial(sdpa, implementation=None)
    sdpa_ans = partial(sdpa, implementation=impl)
    if use_vmap:
      sdpa_ans = jax.vmap(sdpa_ans, in_axes=(0, 0, 0, None, None), out_axes=0)

    # For testing purposes, we call the non-GQA version without vmap in the
    # reference code
    K_ref = jnp.repeat(K, G, axis=2)
    V_ref = jnp.repeat(V, G, axis=2)
    out_ref, sdpa_vjp_ref = jax.vjp(sdpa_ref, Q, K_ref, V_ref, bias, mask)
    out_ans, sdpa_vjp_ans = jax.vjp(sdpa_ans, Q, K, V, bias, mask)

    dQ_ref, dK_ref, dV_ref = sdpa_vjp_ref(grad)[:3]
    dQ_ans, dK_ans, dV_ans = sdpa_vjp_ans(grad)[:3]
    dK_ref = dK_ref.reshape(B, S, N // G, G, H).sum(axis=3)
    dV_ref = dV_ref.reshape(B, S, N // G, G, H).sum(axis=3)

    if impl == 'cudnn':
      self.assertTrue(_check_cudnn_backend(sdpa_ans, Q, K, V, bias, mask))
      self.assertTrue(_check_cudnn_backend(sdpa_vjp_ans, grad))

    self.assertAllClose(out_ref, out_ans, atol=.01, rtol=.01)
    self.assertAllClose(dQ_ref, dQ_ans, rtol=.01, atol=.01)
    self.assertAllClose(dK_ref, dK_ans, rtol=.01, atol=.01)
    self.assertAllClose(dV_ref, dV_ans, rtol=.01, atol=.01)

  @parameterized.product(
      mask_mode=['bias', 'causal', 'padding', 'custom', ('causal', 'padding'),
                 ('custom', 'padding'), ('bias', 'causal'),
                 ('causal', 'sliding_window')],
  )
  def testDotProductAttentionMask(self, mask_mode):
    if isinstance(mask_mode, str):
      mask_mode = (mask_mode,)
    min_cudnn_version = 90200 if 'sliding_window' in mask_mode else 8904
    if not _is_required_cudnn_version_satisfied(min_cudnn_version):
      raise unittest.SkipTest("CUDA or cuDNN versions are not compatible.")

    dtype = jnp.bfloat16
    B, S, T, N, H = 2, 128, 128, 4, 32
    keys = random.split(random.PRNGKey(0), 4)
    Q = random.normal(keys[0], (B, T, N, H), dtype)
    K = random.normal(keys[1], (B, S, N, H), dtype)
    V = random.normal(keys[2], (B, S, N, H), dtype)
    grad = random.normal(keys[3], (B, T, N, H), dtype)
    bias, mask = None, None
    q_seqlen, kv_seqlen = None, None
    window_size = None

    is_causal = 'causal' in mask_mode
    if 'padding' in mask_mode:
      q_seqlen = jnp.array([T // 2, T // 4], dtype=jnp.int32)
      kv_seqlen = jnp.array([S // 4, S // 2], dtype=jnp.int32)
    if 'custom' in mask_mode:
      # Use a generated causal mask as the custom mask.
      custom_mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
      mask = custom_mask[None, None, :, :]
    if 'bias' in mask_mode:
      bias = random.normal(keys[4], (1, N, T, S), dtype)
    if 'sliding_window' in mask_mode:
      window_size = (3, 2) if is_causal else (3, 0)

    sdpa = nn.dot_product_attention
    sdpa_ref = partial(sdpa, is_causal=is_causal, implementation=None)
    sdpa_ans = partial(sdpa, is_causal=is_causal, implementation='cudnn')

    args = (Q, K, V, bias, mask)
    kwargs = {'query_seq_lengths': q_seqlen, 'key_value_seq_lengths': kv_seqlen}

    # Convert the kargs to positional args for the jax.vjp.
    fn_ref = lambda q, k, v, b, m, qs, kvs: sdpa_ref(
        q, k, v, b, m, query_seq_lengths=qs, key_value_seq_lengths=kvs,
        local_window_size=window_size,
    )
    fn_ans = lambda q, k, v, b, m, qs, kvs: sdpa_ans(
        q, k, v, b, m, query_seq_lengths=qs, key_value_seq_lengths=kvs,
        local_window_size=window_size,
    )
    out_ref, sdpa_vjp_ref = jax.vjp(fn_ref, *args, q_seqlen, kv_seqlen)
    out_ans, sdpa_vjp_ans = jax.vjp(fn_ans, *args, q_seqlen, kv_seqlen)
    dQ_ref, dK_ref, dV_ref, dbias_ref = sdpa_vjp_ref(grad)[:4]
    dQ_ans, dK_ans, dV_ans, dbias_ans = sdpa_vjp_ans(grad)[:4]

    # Check if cudnn backend is called.
    self.assertTrue(_check_cudnn_backend(sdpa_ans, *args, **kwargs))
    self.assertTrue(_check_cudnn_backend(sdpa_vjp_ans, grad))

    self.assertAllClose(out_ref, out_ans, atol=.01, rtol=.01)
    self.assertAllClose(dQ_ref, dQ_ans, rtol=.02, atol=.02)
    self.assertAllClose(dK_ref, dK_ans, rtol=.02, atol=.02)
    self.assertAllClose(dV_ref, dV_ans, rtol=.01, atol=.01)
    self.assertAllClose(dbias_ref, dbias_ans, rtol=.02, atol=.02)

  @parameterized.product(
      batch_size=[1, 16],
      use_vmap=[False, True],
  )
  def testDotProductAttentionBiasGradient(self, batch_size, use_vmap):
    if not _is_required_cudnn_version_satisfied(8904):
      raise unittest.SkipTest("CUDA or cuDNN versions are not compatible.")

    dtype = jnp.bfloat16
    B, S, N, H = batch_size, 128, 4, 32
    keys = random.split(random.PRNGKey(0), 2)
    x = random.normal(keys[0], (B, S, N, H), dtype)
    bias = random.normal(keys[1], (B, N, S, S), dtype=dtype)
    mask = jnp.ones((1, 1, S), dtype=jnp.bool_)

    def attention(x, bias, mask, impl):
      return jax.nn.dot_product_attention(
          query=x,
          key=x,
          value=x,
          bias=bias,
          mask=mask,
          is_causal=False,
          implementation=impl,
      )
    attn_ref = partial(attention, impl=None)
    attn_ans = partial(attention, impl='cudnn')
    if use_vmap:
      attn_batched_ref = jax.vmap(attn_ref, in_axes=(0, 0, None))
      attn_batched_ans = jax.vmap(attn_ans, in_axes=(0, 0, None))
    else:
      attn_batched_ref = attn_ref
      attn_batched_ans = attn_ans

    fwd_ref = jax.jit(attn_batched_ref)
    fwd_ans = jax.jit(attn_batched_ans)
    y_ref = fwd_ref(x, bias, mask)
    y_ans = fwd_ans(x, bias, mask)
    self.assertAllClose(y_ref, y_ans)

    @jax.jit
    def bwd_ref(x, bias, mask):
      _, f_vjp = jax.vjp(attn_ref, x, bias, mask)
      return f_vjp(x)
    @jax.jit
    def bwd_ans(x, bias, mask):
      _, f_vjp = jax.vjp(attn_ans, x, bias, mask)
      return f_vjp(x)

    if batch_size != 1:
      with self.assertRaisesRegex(ValueError, _cudnn_dbias_error):
        _, dbias_ans, _ = bwd_ans(x, bias, mask)
    else:
      _, dbias_ref, _ = bwd_ref(x, bias, mask)
      _, dbias_ans, _ = bwd_ans(x, bias, mask)
      self.assertAllClose(dbias_ans, dbias_ref, rtol=0.1, atol=0.1)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testSoftplusGrad(self):
    check_grads(nn.softplus, (1e-8,), order=4,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSoftplusGradZero(self):
    check_grads(nn.softplus, (0.,), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSoftplusGradInf(self):
    self.assertAllClose(
        1., jax.grad(nn.softplus)(float('inf')))

  def testSoftplusGradNegInf(self):
    check_grads(nn.softplus, (-float('inf'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSoftplusGradNan(self):
    check_grads(nn.softplus, (float('nan'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  @parameterized.parameters([int, float] + jtu.dtypes.floating + jtu.dtypes.integer)
  def testSoftplusZero(self, dtype):
    self.assertEqual(jnp.log(dtype(2)), nn.softplus(dtype(0)))

  def testSparseplusGradZero(self):
    check_grads(nn.sparse_plus, (-2.,), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSparseplusGrad(self):
    check_grads(nn.sparse_plus, (0.,), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSparseplusAndSparseSigmoid(self):
    self.assertAllClose(
        jax.grad(nn.sparse_plus)(0.), nn.sparse_sigmoid(0.),
        check_dtypes=False)
    self.assertAllClose(
        jax.grad(nn.sparse_plus)(2.), nn.sparse_sigmoid(2.),
        check_dtypes=False)
    self.assertAllClose(
        jax.grad(nn.sparse_plus)(-2.), nn.sparse_sigmoid(-2.),
        check_dtypes=False)

  def testSquareplusGrad(self):
    check_grads(nn.squareplus, (1e-8,), order=4,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSquareplusGradZero(self):
    check_grads(nn.squareplus, (0.,), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSquareplusGradNegInf(self):
    check_grads(nn.squareplus, (-float('inf'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testSquareplusGradNan(self):
    check_grads(nn.squareplus, (float('nan'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  @parameterized.parameters([float] + jtu.dtypes.floating)
  def testSquareplusZero(self, dtype):
    self.assertEqual(dtype(1), nn.squareplus(dtype(0), dtype(4)))

  def testMishGrad(self):
    check_grads(nn.mish, (1e-8,), order=4,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testMishGradZero(self):
    check_grads(nn.mish, (0.,), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testMishGradNegInf(self):
    check_grads(nn.mish, (-float('inf'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  def testMishGradNan(self):
    check_grads(nn.mish, (float('nan'),), order=1,
                rtol=1e-2 if jtu.test_device_matches(["tpu"]) else None)

  @parameterized.parameters([float] + jtu.dtypes.floating)
  def testMishZero(self, dtype):
    self.assertEqual(dtype(0), nn.mish(dtype(0)))

  def testReluGrad(self):
    rtol = 1e-2 if jtu.test_device_matches(["tpu"]) else None
    check_grads(nn.relu, (1.,), order=3, rtol=rtol)
    check_grads(nn.relu, (-1.,), order=3, rtol=rtol)
    jaxpr = jax.make_jaxpr(jax.grad(nn.relu))(0.)
    self.assertGreaterEqual(len(jaxpr.jaxpr.eqns), 2)

  def testReluGradAtZero(self):
    # https://dl.acm.org/doi/10.5555/3540261.3540297
    grad = jax.grad(nn.relu)(0.)
    self.assertEqual(grad, 0.)

  def testRelu6Grad(self):
    rtol = 1e-2 if jtu.test_device_matches(["tpu"]) else None
    check_grads(nn.relu6, (1.,), order=3, rtol=rtol)
    check_grads(nn.relu6, (-1.,), order=3, rtol=rtol)
    self.assertAllClose(jax.grad(nn.relu6)(0.), 0., check_dtypes=False)
    self.assertAllClose(jax.grad(nn.relu6)(6.), 0., check_dtypes=False)

  def testSoftplusValue(self):
    val = nn.softplus(89.)
    self.assertAllClose(val, 89., check_dtypes=False)

  def testSparseplusValue(self):
    val = nn.sparse_plus(89.)
    self.assertAllClose(val, 89., check_dtypes=False)

  def testSparsesigmoidValue(self):
    self.assertAllClose(nn.sparse_sigmoid(-2.), 0., check_dtypes=False)
    self.assertAllClose(nn.sparse_sigmoid(2.), 1., check_dtypes=False)
    self.assertAllClose(nn.sparse_sigmoid(0.), .5, check_dtypes=False)

  def testSquareplusValue(self):
    val = nn.squareplus(1e3)
    self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

  def testMishValue(self):
    val = nn.mish(1e3)
    self.assertAllClose(val, 1e3, check_dtypes=False, atol=1e-3)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testEluGrad(self):
    check_grads(nn.elu, (1e4,), order=4, eps=1.)

  def testEluValue(self):
    val = nn.elu(1e4)
    self.assertAllClose(val, 1e4, check_dtypes=False)

  def testGluValue(self):
    val = nn.glu(jnp.array([1.0, 0.0]), axis=0)
    self.assertAllClose(val, jnp.array([0.5]))

  @parameterized.parameters(False, True)
  def testGeluIntType(self, approximate):
    val_float = nn.gelu(jnp.array(-1.0), approximate=approximate)
    val_int = nn.gelu(jnp.array(-1), approximate=approximate)
    self.assertAllClose(val_float, val_int)

  @parameterized.parameters(False, True)
  def testGelu(self, approximate):
    def gelu_reference(x):
      return x * scipy.stats.norm.cdf(x)
    args_maker = lambda: [jnp.linspace(-12, 5, 10000, dtype=jnp.float32)]
    rtol = 2e-5
    atol = 1e-3 if approximate else 0
    self._CheckAgainstNumpy(
        gelu_reference,
        partial(nn.gelu, approximate=approximate),
        args_maker,
        check_dtypes=False,
        tol=0,
        rtol=rtol,
        atol=atol,
    )

  @parameterized.parameters(*itertools.product(
      (jnp.float32, jnp.bfloat16, jnp.float16),
      (partial(nn.gelu, approximate=False),
       partial(nn.gelu, approximate=True),
       nn.relu, nn.softplus, nn.sparse_plus, nn.sigmoid, nn.squareplus, nn.mish)))
  def testDtypeMatchesInput(self, dtype, fn):
    x = jnp.zeros((), dtype=dtype)
    out = fn(x)
    self.assertEqual(out.dtype, dtype)

  def testEluMemory(self):
    # see https://github.com/jax-ml/jax/pull/1640
    with jax.enable_checks(False):  # With checks we materialize the array
      jax.make_jaxpr(lambda: nn.elu(jnp.ones((10 ** 12,))))  # don't oom

  def testHardTanhMemory(self):
    # see https://github.com/jax-ml/jax/pull/1640
    with jax.enable_checks(False):  # With checks we materialize the array
      jax.make_jaxpr(lambda: nn.hard_tanh(jnp.ones((10 ** 12,))))  # don't oom

  @parameterized.parameters([nn.softmax, nn.log_softmax])
  def testSoftmaxEmptyArray(self, fn):
    x = jnp.array([], dtype=float)
    self.assertArraysEqual(fn(x), x)

  @parameterized.parameters([nn.softmax, nn.log_softmax])
  def testSoftmaxEmptyMask(self, fn):
    x = jnp.array([5.5, 1.3, -4.2, 0.9])
    m = jnp.zeros_like(x, dtype=bool)
    expected = jnp.full_like(x, 0.0 if fn is nn.softmax else -jnp.inf)
    self.assertArraysEqual(fn(x, where=m), expected)

  @parameterized.parameters([nn.softmax, nn.log_softmax])
  def testSoftmaxWhereMask(self, fn):
    x = jnp.array([5.5, 1.3, -4.2, 0.9])
    m = jnp.array([True, False, True, True])

    out = fn(x, where=m)
    self.assertAllClose(out[m], fn(x[m]))

    probs = out if fn is nn.softmax else jnp.exp(out)
    self.assertAllClose(probs.sum(), 1.0)

    # TODO(mattjj): include log_softmax in these extra tests if/when we add a
    # custom_jvp rule for it (since otherwise it doesn't pass the numerical
    # checks below).
    if fn is nn.softmax and config.softmax_custom_jvp.value:
      g_fun = lambda x: jnp.take(fn(x, where=m, initial=-jnp.inf),
                                jnp.array([0, 2, 3]))
      jtu.check_grads(g_fun, (x,), order=2)

  @parameterized.parameters([nn.softmax, nn.log_softmax])
  def testSoftmaxWhereGrad(self, fn):
    # regression test for https://github.com/jax-ml/jax/issues/19490
    x = jnp.array([36., 10000.])
    mask = x < 1000

    f = lambda x, mask: fn(x, where=mask)[0]

    self.assertAllClose(jax.grad(f)(x, mask), jnp.zeros_like(x))

  def testSoftmaxGrad(self):
    x = jnp.array([5.5, 1.3, -4.2, 0.9])
    jtu.check_grads(nn.softmax, (x,), order=2, atol=5e-3)

  def testSoftmaxGradResiduals(self):
    if not config.softmax_custom_jvp.value:
      raise unittest.SkipTest("only applies when upgrade flag enabled")
    x = jnp.array([5.5, 1.3, -4.2, 0.9])
    res = ad_checkpoint.saved_residuals(nn.softmax, x)
    self.assertLen(res, 1)

  def testSoftmaxGradFlag(self):
    x = jnp.array([5.5, 1.3, -4.2, 0.9])

    with jax.softmax_custom_jvp(False):
      res = ad_checkpoint.saved_residuals(nn.softmax, x)
    self.assertLen(res, 3)
    self.assertEqual(sum(a.size for a, _ in res), 6)

    with jax.softmax_custom_jvp(True):
      res = ad_checkpoint.saved_residuals(nn.softmax, x)
    self.assertLen(res, 1)
    self.assertEqual(sum(a.size for a, _ in res), 4)

  def testStandardizeWhereMask(self):
    x = jnp.array([5.5, 1.3, -4.2, 0.9])
    m = jnp.array([True, False, True, True])
    x_filtered = jnp.take(x, jnp.array([0, 2, 3]))

    out_masked = jnp.take(nn.standardize(x, where=m), jnp.array([0, 2, 3]))
    out_filtered = nn.standardize(x_filtered)

    self.assertAllClose(out_masked, out_filtered)

  def testOneHot(self):
    actual = nn.one_hot(jnp.array([0, 1, 2]), 3)
    expected = jnp.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])
    self.assertAllClose(actual, expected, check_dtypes=False)

    actual = nn.one_hot(jnp.array([1, 2, 0]), 3)
    expected = jnp.array([[0., 1., 0.],
                          [0., 0., 1.],
                          [1., 0., 0.]])
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testOneHotOutOfBound(self):
    actual = nn.one_hot(jnp.array([-1, 3]), 3)
    expected = jnp.array([[0., 0., 0.],
                          [0., 0., 0.]])
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testOneHotNonArrayInput(self):
    actual = nn.one_hot([0, 1, 2], 3)
    expected = jnp.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testOneHotCustomDtype(self):
    actual = nn.one_hot(jnp.array([0, 1, 2]), 3, dtype=jnp.bool_)
    expected = jnp.array([[True, False, False],
                          [False, True, False],
                          [False, False, True]])
    self.assertAllClose(actual, expected)

  def testOneHotConcretizationError(self):
    # https://github.com/jax-ml/jax/issues/3654
    msg = r"in jax.nn.one_hot argument `num_classes`"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      jax.jit(nn.one_hot)(3, 5)

  def testOneHotAxis(self):
    expected = jnp.array([[0., 1., 0.],
                          [0., 0., 1.],
                          [1., 0., 0.]]).T

    actual = nn.one_hot(jnp.array([1, 2, 0]), 3, axis=0)
    self.assertAllClose(actual, expected, check_dtypes=False)

    actual = nn.one_hot(jnp.array([1, 2, 0]), 3, axis=-2)
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testOneHotNonInteger(self):
    with self.assertDeprecationWarnsOrRaises("jax-nn-one-hot-float-input",
                                             "jax.nn.one_hot input should be integer-typed"):
      nn.one_hot(jnp.array([1.0]), 3)

  def testTanhExists(self):
    nn.tanh  # doesn't crash

  def testCustomJVPLeak(self):
    # https://github.com/jax-ml/jax/issues/8171
    @jax.jit
    def fwd():
      a = jnp.array(1.)

      def f(hx, _):
        hx = jax.nn.sigmoid(hx + a)
        return hx, None

      hx = jnp.array(0.)
      jax.lax.scan(f, hx, None, length=2)

    with jax.checking_leaks():
      fwd()  # doesn't crash

  def testCustomJVPLeak2(self):
    # https://github.com/jax-ml/jax/issues/8171
    # The above test uses jax.nn.sigmoid, as in the original #8171, but that
    # function no longer actually has a custom_jvp! So we inline the old def.

    @jax.custom_jvp
    def sigmoid(x):
      one = jnp.float32(1)
      return jax.lax.div(one, jax.lax.add(one, jax.lax.exp(jax.lax.neg(x))))
    sigmoid.defjvps(lambda g, ans, x: g * ans * (jnp.float32(1) - ans))

    @jax.jit
    def fwd():
      a = jnp.array(1., 'float32')

      def f(hx, _):
        hx = sigmoid(hx + a)
        return hx, None

      hx = jnp.array(0., 'float32')
      jax.lax.scan(f, hx, None, length=2)

    with jax.checking_leaks():
      fwd()  # doesn't crash


InitializerRecord = collections.namedtuple(
  "InitializerRecord",
  ["name", "initializer", "shapes", "dtypes"])

ALL_SHAPES = [(2,), (2, 2), (2, 3), (3, 2), (2, 3, 4), (4, 3, 2), (2, 3, 4, 5)]

def initializer_record(name, initializer, dtypes, min_dims=2, max_dims=4):
  shapes = [shape for shape in ALL_SHAPES
            if min_dims <= len(shape) <= max_dims]
  return InitializerRecord(name, initializer, shapes, dtypes)

INITIALIZER_RECS = [
    initializer_record("uniform", nn.initializers.uniform, jtu.dtypes.floating, 1),
    initializer_record("normal", nn.initializers.normal, jtu.dtypes.inexact, 1),
    initializer_record("he_normal", nn.initializers.he_normal, jtu.dtypes.inexact),
    initializer_record("he_uniform", nn.initializers.he_uniform, jtu.dtypes.inexact),
    initializer_record("glorot_normal", nn.initializers.glorot_normal, jtu.dtypes.inexact),
    initializer_record("glorot_uniform", nn.initializers.glorot_uniform, jtu.dtypes.inexact),
    initializer_record("lecun_normal", nn.initializers.lecun_normal, jtu.dtypes.inexact),
    initializer_record("lecun_uniform", nn.initializers.lecun_uniform, jtu.dtypes.inexact),
    initializer_record("orthogonal", nn.initializers.orthogonal, jtu.dtypes.floating, 2, 2),
    initializer_record("truncated_normal", nn.initializers.truncated_normal, jtu.dtypes.floating, 1),
    initializer_record("delta_orthogonal", nn.initializers.delta_orthogonal, jtu.dtypes.floating, 4, 4)
]


@jtu.with_config(jax_legacy_prng_key="allow")
class NNInitializersTest(jtu.JaxTestCase):
  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(initializer=rec.initializer())],
      shape=rec.shapes,
      dtype=rec.dtypes
    )
    for rec in INITIALIZER_RECS
  ))
  def testInitializer(self, initializer, shape, dtype):
    rng = random.PRNGKey(0)
    val = initializer(rng, shape, dtype)

    self.assertEqual(shape, jnp.shape(val))
    self.assertEqual(jax.dtypes.canonicalize_dtype(dtype), jnp.dtype(val))

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(initializer_provider=rec.initializer)],
      shape=rec.shapes,
      dtype=rec.dtypes
    )
    for rec in INITIALIZER_RECS
  ))
  def testInitializerProvider(self, initializer_provider, shape, dtype):
    rng = random.PRNGKey(0)
    initializer = initializer_provider(dtype=dtype)
    val = initializer(rng, shape)

    self.assertEqual(shape, jnp.shape(val))
    self.assertEqual(jax.dtypes.canonicalize_dtype(dtype), jnp.dtype(val))

  def testVarianceScalingMultiAxis(self):
    rng = random.PRNGKey(0)
    shape = (2, 3, 4, 5)
    initializer = nn.initializers.variance_scaling(
      scale=1.0, mode='fan_avg', distribution='truncated_normal',
      in_axis=(0, 1), out_axis=(-2, -1))
    val = initializer(rng, shape)

    self.assertEqual(shape, jnp.shape(val))

  def testVarianceScalingBatchAxis(self):
    rng = random.PRNGKey(0)
    shape = (2, 3, 4, 5)
    initializer = nn.initializers.variance_scaling(
      scale=1.0, mode='fan_avg', distribution='truncated_normal',
      in_axis=0, out_axis=(2, 3), batch_axis=1)
    val = initializer(rng, shape)

    self.assertEqual(shape, jnp.shape(val))

  def testVarianceScalingError(self):
    rng = random.PRNGKey(0)
    shape = (5,)
    initializer = nn.initializers.variance_scaling(
      scale=1.0, mode='fan_avg', distribution='truncated_normal')

    with self.assertRaisesRegex(
      ValueError,
      "Can't compute input and output sizes of a 1"
      "-dimensional weights tensor. Must be at least 2D."
    ):
      initializer(rng, shape)

  def testAccidentalUpcasting(self):
    rng = random.PRNGKey(0)
    shape = (4, 4)
    scalar_param = jnp.array(1.0, dtype=jnp.float32)
    for init_fn in (nn.initializers.uniform(scalar_param, jnp.bfloat16),
                    nn.initializers.normal(scalar_param, jnp.bfloat16),
                    nn.initializers.truncated_normal(scalar_param, jnp.bfloat16),
                   ):
      sub_rng, rng = random.split(rng)
      val = init_fn(sub_rng, shape)
      self.assertEqual(val.dtype, jnp.bfloat16)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
