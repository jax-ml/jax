from __future__ import absolute_import, division, print_function
import numpy as onp
import jax.numpy as np
from jax.numpy import linalg
from jax import random

def get_squares_():
  key = random.PRNGKey(42)
  # key, *subkey = random.split(key, )
  values = np.tile(random.uniform(key, shape=100, minval=0, maxval=100), 10)
  return values

class Linalg:
    params = ['svd', 'pinv', 'det', 'norm']
    param_names = ['op']

    def setup(self, op, typename):
        np.seterr(all='ignore')

        self.func = getattr(linalg, op)

        if op == 'cholesky':
            # we need a positive definite
            self.a = np.dot(get_squares_(),
                            get_squares_().T)
        else:
            self.a = get_squares_()

    def time_op(self, op, typename):
        self.func(self.a)
        print(op)
        print(self.a)
        print(self.func.op)