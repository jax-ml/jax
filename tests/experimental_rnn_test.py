# Copyright 2022 The JAX Authors.
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
from functools import partial
from absl.testing import absltest
import numpy as np
import jax
import jax.numpy as jnp
from jax._src import lib
from jax._src import test_util as jtu
from jax.experimental import rnn

from jax._src.config import config

config.parse_flags_with_absl()


class RnnTest(jtu.JaxTestCase):

  @jtu.sample_product(
      batch_size=[1, 4],
      seq_len=[1, 4],
      input_size=[1, 2],
      hidden_size=[1, 6],
      num_layers=[1, 4],
      bidirectional=[True, False],
  )
  @jtu.skip_on_devices("cpu", "tpu", "rocm")
  def test_lstm(self, batch_size: int, seq_len: int, input_size: int,
                hidden_size: int, num_layers: int, bidirectional: bool):
    num_directions = 2 if bidirectional else 1

    seq_lengths = jax.random.randint(
      jax.random.PRNGKey(0), (batch_size,), 0, seq_len, dtype=jnp.int32)

    root_key = jax.random.PRNGKey(1)
    k1, k2, k3, k4 = jax.random.split(root_key, 4)
    x = jax.random.normal(
        k1, (batch_size, seq_len, input_size), dtype=jnp.float32)
    h_0 = jax.random.normal(
        k2, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    c_0 = jax.random.normal(
        k3, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    weights = rnn.init_lstm_weight(k4, input_size, hidden_size, num_layers,
                                   bidirectional)
    @partial(jax.value_and_grad, has_aux=True)
    def f(weights, x, h_0, c_0):
      y, h, c = rnn.lstm(
        x,
        h_0,
        c_0,
        weights,
        seq_lengths=seq_lengths,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=False,
        bidirectional=bidirectional)

      loss = jnp.sum(y)
      return loss, (y, h, c)



    (loss, (y, h_n, c_n)), grad = f(weights, x, h_0, c_0)
    jtu.check_grads(f, (x, h_0, c_0, weights), modes=['rev'], order=1)

    self.assertFalse(np.isnan(loss))
    self.assertFalse(np.isnan(grad).any())

    @partial(jax.value_and_grad, has_aux=True)
    def g(weights, x, h_0, c_0):
      W_ih, W_hh, b_ih, b_hh = rnn.unpack_lstm_weights(weights, input_size,
                                                      hidden_size, num_layers,
                                                      bidirectional)
      y_ref, h_n_ref, c_n_ref = rnn.lstm_ref(
          x,
          h_0,
          c_0,
          W_ih,
          W_hh,
          b_ih,
          b_hh,
          seq_lengths=seq_lengths,
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          dropout=False,
          bidirectional=bidirectional)

      loss = jnp.sum(y_ref)
      return loss, (y_ref, h_n_ref, c_n_ref)

    (loss_ref, (y_ref, h_n_ref, c_n_ref)), grad_ref = g(weights, x, h_0, c_0)

    self.assertFalse(np.isnan(loss_ref))
    self.assertFalse(np.isnan(grad_ref).any())

    np.testing.assert_allclose(y_ref, y, rtol=1e-05, atol=1e-5)
    np.testing.assert_allclose(h_n_ref, h_n, rtol=1e-05, atol=1e-5)
    np.testing.assert_allclose(c_n_ref, c_n, rtol=1e-05, atol=1e-5)

  @jtu.sample_product(
    batch_size=[1, 4],
    seq_len=[1, 4],
    input_size=[1, 2],
    hidden_size=[1, 6],
    num_layers=[1, 4],
    bidirectional=[True, False],
  )
  def test_lstm_ref(self, batch_size: int, seq_len: int, input_size: int,
                    hidden_size: int, num_layers: int, bidirectional: bool):

    num_directions = 2 if bidirectional else 1

    seq_lengths = jax.random.randint(
      jax.random.PRNGKey(0), (batch_size,), 0, seq_len, dtype=jnp.int32)

    root_key = jax.random.PRNGKey(1)
    k1, k2, k3, k4 = jax.random.split(root_key, 4)
    x = jax.random.normal(
        k1, (batch_size, seq_len, input_size), dtype=jnp.float32)
    h_0 = jax.random.normal(
        k2, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    c_0 = jax.random.normal(
        k3, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    weights = rnn.init_lstm_weight(k4, input_size, hidden_size, num_layers,
                                   bidirectional)

    @partial(jax.value_and_grad, has_aux=True)
    def f(weights, x, h_0, c_0):
      W_ih, W_hh, b_ih, b_hh = rnn.unpack_lstm_weights(weights, input_size,
                                                      hidden_size, num_layers,
                                                      bidirectional)
      y_ref, h_n_ref, c_n_ref = rnn.lstm_ref(
          x,
          h_0,
          c_0,
          W_ih,
          W_hh,
          b_ih,
          b_hh,
          seq_lengths=seq_lengths,
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          dropout=False,
          bidirectional=bidirectional)

      loss = jnp.sum(y_ref)

      return loss, (y_ref, h_n_ref, c_n_ref)

    (loss_ref, (y_ref, h_n_ref, c_n_ref)), grad_ref = f(weights, x, h_0, c_0)

    self.assertFalse(np.isnan(loss_ref))
    self.assertFalse(np.isnan(grad_ref).any())

    self.assertEqual(y_ref.shape, (batch_size, seq_len, num_directions * hidden_size))

    for i in range(batch_size):
      y_padded = y_ref[i, seq_lengths[i]:]
      np.testing.assert_allclose(y_padded, jnp.zeros_like(y_padded))

  @jtu.skip_on_devices("cpu", "tpu", "rocm")
  def test_lstm_with_varying_seq_lens(self):
    if lib.version < (0, 4, 7):
      # TODO(sharadmv, zhangqiaorjc): remove this when minimum jaxlib version is
      # bumped
      self.skipTest("Need latest jaxlib for this test to pass.")
    batch_size = 6
    seq_len = 7
    input_size = 8
    hidden_size = 12
    num_layers = 5
    bidirectional = False
    num_directions = 2 if bidirectional else 1

    seq_lengths = jnp.array([4, 5, 1, 1, 1, 1], dtype=jnp.dtype("int32"))

    root_key = jax.random.PRNGKey(1)
    k1, k2, k3, k4 = jax.random.split(root_key, 4)
    x = jax.random.normal(
        k1, (batch_size, seq_len, input_size), dtype=jnp.float32)
    h_0 = jax.random.normal(
        k2, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    c_0 = jax.random.normal(
        k3, (num_directions * num_layers, batch_size, hidden_size),
        dtype=jnp.float32)
    weights = rnn.init_lstm_weight(k4, input_size, hidden_size, num_layers,
                                   bidirectional)

    @jax.jit
    def f(x, h_0, c_0, weights):
      return rnn.lstm(
          x,
          h_0,
          c_0,
          weights,
          seq_lengths=seq_lengths,
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          dropout=False,
          bidirectional=bidirectional)

    jtu.check_grads(f, (x, h_0, c_0, weights), modes=['rev'], order=1)

    # TODO(sharadmv): enable when lstm_ref works with seq_lengths
    # W_ih, W_hh, b_ih, b_hh = rnn.unpack_lstm_weights(weights, input_size,
    #                                                  hidden_size, num_layers,
    #                                                  bidirectional)
    # y_ref, h_n_ref, c_n_ref = rnn.lstm_ref(
    #     x,
    #     h_0,
    #     c_0,
    #     W_ih,
    #     W_hh,
    #     b_ih,
    #     b_hh,
    #     seq_lengths=seq_lengths,
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     num_layers=num_layers,
    #     dropout=False,
    #     bidirectional=bidirectional)

    # np.testing.assert_allclose(y_ref, y, rtol=1e-05, atol=1e-5)
    # np.testing.assert_allclose(h_n_ref, h_n, rtol=1e-05, atol=1e-5)
    # np.testing.assert_allclose(c_n_ref, c_n, rtol=1e-05, atol=1e-5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
