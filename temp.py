from jax import make_jaxpr
import jax.numpy as np

def f(x, y):
  return (np.sin(x), np.cos(x) * y, np.tanh(y))
  # return (2. * np.sin(x), 3. * np.cos(x) * y, np.tanh(y), 4.)

jaxpr = make_jaxpr(f)(2., 3.)


import jax.interpreters.partial_eval as pe
from jax.abstract_arrays import ShapedArray
from jax.core import AbstractTuple

avals = (AbstractTuple(()),
         ShapedArray((), np.float32),
         ShapedArray((), np.float32))
pe.partial_eval_jaxpr(jaxpr, avals, ((), True, False))
