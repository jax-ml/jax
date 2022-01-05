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
