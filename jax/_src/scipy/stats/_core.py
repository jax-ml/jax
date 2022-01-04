import jax.numpy as jnp
from jax import vmap

def mode(x, axis=None):

  """
    Return an array of the modal (most common) value in the passed array.    
  """
  def _mode(x):
    vals, counts = jnp.unique(x, return_counts=True, size=x.size)
    return vals[jnp.argmax(counts)]
  if axis is None:
    return _mode(x)
  else:
    x = jnp.moveaxis(x, axis, 0)
    return vmap(_mode, in_axes=(1,))(x.reshape(x.shape[0], -1)).reshape(x.shape[1:])
