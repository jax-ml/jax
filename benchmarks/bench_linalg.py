from __future__ import absolute_import, division, print_function
import jax.numpy as np
from jax.numpy import linalg
from jax import random

class Eindot():
  def setup(self):
    self.a = np.arange(60000.0).reshape(150, 400)
    self.ac = self.a.copy()
    self.at = self.a.T
    self.atc = self.a.T.copy()
    self.b = np.arange(240000.0).reshape(400, 600)
    self.c = np.arange(600)
    self.d = np.arange(400)

    self.a3 = np.arange(480000.).reshape(60, 80, 100)
    self.b3 = np.arange(192000.).reshape(80, 60, 40)

  def time_dot_a_b(self):
    np.dot(self.a, self.b)

  def time_dot_d_dot_b_c(self):
    np.dot(self.d, np.dot(self.b, self.c))

  def time_dot_trans_a_at(self):
    np.dot(self.a, self.at)

  def time_dot_trans_a_atc(self):
    np.dot(self.a, self.atc)

  def time_dot_trans_at_a(self):
    np.dot(self.at, self.a)

  def time_dot_trans_atc_a(self):
    np.dot(self.atc, self.a)

  def time_einsum_i_ij_j(self):
    np.einsum('i,ij,j', self.d, self.b, self.c)

  def time_einsum_ij_jk_a_b(self):
    np.einsum('ij,jk', self.a, self.b)

  def time_einsum_ijk_jil_kl(self):
    np.einsum('ijk,jil->kl', self.a3, self.b3)

  def time_inner_trans_a_a(self):
    np.inner(self.a, self.a)

  def time_inner_trans_a_ac(self):
    np.inner(self.a, self.ac)

  def time_matmul_a_b(self):
    np.matmul(self.a, self.b)

  def time_matmul_d_matmul_b_c(self):
    np.matmul(self.d, np.matmul(self.b, self.c))

  def time_matmul_trans_a_at(self):
    np.matmul(self.a, self.at)

  def time_matmul_trans_a_atc(self):
    np.matmul(self.a, self.atc)

  def time_matmul_trans_at_a(self):
    np.matmul(self.at, self.a)

  def time_matmul_trans_atc_a(self):
    np.matmul(self.atc, self.a)

  def time_tensordot_a_b_axes_1_0_0_1(self):
    np.tensordot(self.a3, self.b3, axes=([1, 0], [0, 1]))

def get_squares_(size):
  key = random.PRNGKey(42)
  values = random.uniform(key, shape=(size, size), minval=0, maxval=100)
  return values

class Linalg:
  params = [['svd', 'pinv', 'det', 'norm'], [10, 100, 200]]
  param_names = ['op', 'size']

  def setup(self, op, size):
    self.func = getattr(linalg, op)
    self.a = get_squares_(size)

  def time_op(self, op, size):
    self.func(self.a)
