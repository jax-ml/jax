# Copyright 2020 Google LLC
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

import functools
import numpy as np
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Sequence

from jax import core
from jax import dtypes
from jax import lax
from jax import lax_linalg

def to_jax_dtype(tf_dtype):
  if tf_dtype.name == 'bfloat16':
    return dtypes.bfloat16
  return tf_dtype.as_numpy_dtype

Limitation = NamedTuple("Limitation", [ ("primitive_name", str)
                                      , ("error_type", str)
                                      , ("error_string", str)
                                      , ("devices", Tuple[str,...])
                                      ])

NpDType = Any

def categorize(prim: core.Primitive, *args, **kwargs) \
    -> List[Limitation]:
  """
  Given a primitive and a set of parameters one would like to pass to it,
  categorize identifies the potential limitations the call would encounter when
  converted to TF through jax2tf.

  Args:
    prim: the primitive to call.
    args: the arguments to pass to prim.
    kwargs: the keyword arguments to pass to prim.

  Returns:
    A list of limitations
  """
  limitations: List[Limitation] = []
  all_devices = ["CPU", "GPU", "TPU"]

  def _report_failure(error_type: str, msg: str,
                      devs: Sequence[str] = all_devices) -> None:
    limitations.append(Limitation(prim.name, error_type, msg, tuple(devs)))

  def tf_unimpl(np_dtype: NpDType, additional_msg: Optional[str] = None,
                devs: Sequence[str] = all_devices) -> None:
    msg = f"{prim.name} is unimplemented for dtype {np_dtype}"
    if additional_msg:
      msg += '; ' + additional_msg
    _report_failure("Missing TF support", msg, devs=devs)

  def _to_np_dtype(dtype) -> NpDType:
    try:
      dtype = to_jax_dtype(dtype)
    except:
      pass
    return np.dtype(dtype)

  if prim in [lax.min_p, lax.max_p]:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.bool_, np.int8, np.uint16, np.uint32, np.uint64,
                    np.complex64, np.complex128]:
      tf_unimpl(np_dtype)

  if prim in [lax.rem_p, lax.atan2_p]:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.float16, dtypes.bfloat16]:
      # b/158006398: TF kernels are missing for 'rem' and 'atan2'
      tf_unimpl(np_dtype)

  if prim is lax.nextafter_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.float16, dtypes.bfloat16]:
      tf_unimpl(np_dtype)

  if prim is lax_linalg.qr_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.complex64, np.complex128]:
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      # experimental_compile=True breaks for complex types.
      tf_unimpl(np_dtype)

  if prim is lax_linalg.svd_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.float16, dtypes.bfloat16]:
      # TODO: SVD on TPU for bfloat16 seems to work for JAX but fails for TF
      tf_unimpl(np_dtype, devs=["TPU"])
    elif np_dtype in [np.complex64, np.complex128]:
      # TODO: on CPU and GPU "No registered 'Svd' OpKernel for XLA_CPU_JIT
      # devices". Works on JAX because JAX uses a custom implementation
      additional_msg = ("this works on JAX because JAX uses a custom "
                        "implementation")
      tf_unimpl(np_dtype, additional_msg=additional_msg, devs=["CPU", "GPU"])

  if prim is lax.select_and_gather_add_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    devices_and_max_bits = [ (["CPU", "GPU"], 64)
                           , (["TPU"], 32)
                           ]
    for devs, max_bits in devices_and_max_bits:
      if dtypes.finfo(np_dtype).bits * 2 > max_bits:
        # TODO: getting an exception "XLA encountered an HLO for which this
        # rewriting is not implemented"
        tf_unimpl(np_dtype, devs=devs)

  if prim in [lax.add_p, lax.reduce_window_sum_p]:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.uint16, np.uint32, np.uint64]:
      # TODO(bchetioui): tf.math.add is not defined for the above types.
      tf_unimpl(np_dtype)

  if prim is lax.mul_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.uint32, np.uint64]:
      # TODO(bchetioui): tf.math.multiply is not defined for the above types.
      tf_unimpl(np_dtype)

  if prim in [lax.scatter_mul_p, lax.scatter_add_p]:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype == np.complex64:
      tf_unimpl(np_dtype, devs=["TPU"])

  if prim is lax.population_count_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype == np.uint32:
      tf_unimpl(np_dtype)

  return limitations

def prettify(limitations: Sequence[Limitation]) -> str:
  """Constructs a summary .md file based on a list of limitations."""
  limitations = sorted(list(set(limitations)))

  def _pipewrap(columns):
    return '| ' + ' | '.join(columns) + ' |'

  title = '# Primitives with limited support'

  column_names = [ 'Affected primitive'
                 , 'Type of limitation'
                 , 'Description'
                 , 'Devices affected' ]

  table = [column_names, ['---'] * len(column_names)]

  for lim in limitations:
    table.append([ lim.primitive_name
                 , lim.error_type
                 , lim.error_string
                 , ', '.join(lim.devices)
                 ])

  return title + '\n\n' + '\n'.join(line for line in map(_pipewrap, table))

def pprint_limitations(limitations: Sequence[Limitation],
                       output_file: Optional[str] = None) -> None:
  output = prettify(limitations)
  if output_file:
    with open(output_file, 'w') as f:
      f.write(output)
  else:
    print(output)

all_limitations: Sequence[Limitation] = []
pprint_all_limitations = functools.partial(pprint_limitations, all_limitations)

def collect_limitations(prim: core.Primitive, func: Callable) -> Callable:
  """
  Wraps a primitive and its corresponding TF implementation with `categorize`.
  """
  def wrapper(*args, **kwargs):
    global all_limitations
    all_limitations += categorize(prim, *args, **kwargs)
    return func(*args, **kwargs)
  return wrapper
