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

from collections import defaultdict
import datetime
import functools
import numpy as np
from typing import Any, Callable, Collection, Dict, List, NamedTuple, Optional,\
                   Tuple, Sequence, Set

from jax import core
from jax import dtypes
from jax import lax
from jax.experimental.jax2tf.jax2tf import tf_not_yet_impl, tf_impl
from jax.interpreters import xla

def to_jax_dtype(tf_dtype):
  if tf_dtype.name == 'bfloat16':
    return dtypes.bfloat16
  return tf_dtype.as_numpy_dtype

NpDType = Any

Limitation = NamedTuple("Limitation", [ ("primitive_name", str)
                                      , ("error_type", str)
                                      , ("error_string", str)
                                      , ("affected_dtypes", Tuple[NpDType,...])
                                      , ("devices", Tuple[str,...])
                                      ])

CATEGORY_POSSIBLE_INCORRECT_RESULTS = "Possible incorrect results"
CATEGORY_MISSING_TF_SUPPORT = "Missing TF support"

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
                      affected_dtype: Optional[NpDType] = None,
                      devs: Sequence[str] = all_devices) -> None:
    affected_dtypes = (
      tuple([affected_dtype]) if affected_dtype is not None else tuple())
    limitations.append(Limitation(prim.name, error_type, msg,
                                  affected_dtypes, tuple(devs)))

  def tf_unimpl(np_dtype: Optional[NpDType] = None,
                additional_msg: Optional[str] = None,
                devs: Sequence[str] = all_devices) -> None:
    msg = "Primitive is unimplemented in TF"
    if additional_msg:
      msg += '; ' + additional_msg
    _report_failure(CATEGORY_MISSING_TF_SUPPORT, msg, np_dtype, devs=devs)

  def tf_possible_incorrect(np_dtype: Optional[NpDType] = None,
                         msg: str = "",
                         devs: Sequence[str] = all_devices) -> None:
    _report_failure(CATEGORY_POSSIBLE_INCORRECT_RESULTS, msg, np_dtype, devs=devs)

  def _to_np_dtype(dtype) -> NpDType:
    try:
      dtype = to_jax_dtype(dtype)
    except:
      pass
    return np.dtype(dtype)

  if args and args[0] is not core.unit:
    np_dtype = _to_np_dtype(args[0].dtype)
  else:
    np_dtype = None

  if prim is lax.regularized_incomplete_beta_p:
    if np_dtype in [np.float16, dtypes.bfloat16]:
      tf_unimpl(np_dtype)

  if prim in [lax.min_p, lax.max_p, lax.reduce_window_min_p,
              lax.reduce_window_max_p]:
    if np_dtype in [np.bool_, np.int8, np.uint16, np.uint32, np.uint64,
                    np.complex64, np.complex128]:
      tf_unimpl(np_dtype)

  if prim in [lax.rem_p, lax.atan2_p]:
    if np_dtype in [np.float16, dtypes.bfloat16]:
      # b/158006398: TF kernels are missing for 'rem' and 'atan2'
      tf_unimpl(np_dtype)

  if prim is lax.nextafter_p:
    if np_dtype in [np.float16, dtypes.bfloat16]:
      tf_unimpl(np_dtype)

  if prim is lax.linalg.cholesky_p:
    if np_dtype in [np.complex64, np.complex128]:
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      # experimental_compile=True breaks for complex types.
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"))

  if prim is lax.linalg.qr_p:
    if np_dtype in [np.complex64, np.complex128]:
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      # experimental_compile=True breaks for complex types.
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"))

  if prim is lax.linalg.eig_p:
    tf_unimpl(additional_msg=("this is a problem only in compiled mode "
                              "(experimental_compile=True))"))
    compute_left_eigenvectors = kwargs['compute_left_eigenvectors']
    compute_right_eigenvectors = kwargs['compute_right_eigenvectors']
    if compute_left_eigenvectors and compute_right_eigenvectors:
      tf_unimpl(additional_msg=("it is not possible to request both left and "
                                "right eigenvectors for now"))

  if prim is lax.linalg.eigh_p:
    if np_dtype in [np.complex64, np.complex128]:
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      # experimental_compile=True breaks for complex types.
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"))

  if prim is lax.linalg.lu_p:
    if np_dtype == np.complex64:
      tf_unimpl(np_dtype, devs=["TPU"])

  if prim is lax.linalg.triangular_solve_p:
    if np_dtype in [dtypes.bfloat16, np.float16]:
      tf_unimpl(np_dtype)

  if prim is lax.linalg.svd_p:
    if np_dtype in [dtypes.bfloat16]:
      # TODO: SVD on TPU for bfloat16 seems to work for JAX but fails for TF
      tf_unimpl(np_dtype, devs=["TPU"])
    elif np_dtype in [np.complex64, np.complex128]:
      # TODO: on CPU and GPU "No registered 'Svd' OpKernel for XLA_CPU_JIT
      # devices". Works on JAX because JAX uses a custom implementation.
      # There exists a XlaSvd operation that could replace tf.linalg.svd in
      # these cases but complex numbers support is not implemented in XLA yet,
      # and the API of XlaSvd is different than the one in JAX/TF, which also
      # limits its useability (e.g. no full_matrices argument, …).
      additional_msg = ("this works on JAX because JAX uses a custom "
                        "implementation")
      tf_unimpl(np_dtype, additional_msg=additional_msg, devs=["CPU", "GPU"])

  if prim is lax.select_and_scatter_add_p:
    if np_dtype in [np.uint64, np.uint32, np.uint16]:
      tf_unimpl(np_dtype)

  if prim is lax.select_and_gather_add_p:
    # TODO: the conversion is only supported for float16/float32 on CPU/GPU,
    # and float16 on TPU. This is because we do not implement a precision
    # reduction in the case where packing 2 n-bit values together results in
    # more than the maximum number of bits allowed on the platform (64 on
    # CPU/GPU, 32 on TPU). This could be fixed by implementing a variadic
    # reduce_window in tfxla, or we can require the user to reduce the
    # precision of their arrays manually based on the platform they run on.
    devices_and_max_bits = [ (["CPU", "GPU"], 64)
                           , (["TPU"], 32)
                           ]
    for devs, max_bits in devices_and_max_bits:
      if dtypes.finfo(np_dtype).bits * 2 > max_bits:
        # TODO: getting an exception "XLA encountered an HLO for which this
        # rewriting is not implemented"
        tf_unimpl(np_dtype, devs=devs)

  if prim in [lax.add_p, lax.reduce_window_sum_p]:
    if np_dtype in [np.uint16, np.uint32, np.uint64]:
      # TODO(bchetioui): tf.math.add is not defined for the above types.
      tf_unimpl(np_dtype)

  if prim is lax.mul_p:
    if np_dtype in [np.uint32, np.uint64]:
      # TODO(bchetioui): tf.math.multiply is not defined for the above types.
      tf_unimpl(np_dtype)

  if prim is lax.sort_p:
    if np_dtype in [np.complex64, np.complex128]:
      tf_unimpl(np_dtype)
    if np_dtype == np.bool_ and len(args) == 2:
      tf_unimpl(np_dtype, additional_msg=(
        "sorting 2 arrays where the first one is an array of booleans is not "
        "supported for XlaSort"))
    if kwargs["is_stable"]:
      tf_unimpl(additional_msg="stable sort not implemented for XlaSort")
    if kwargs["dimension"] != len(np.shape(args[0])) - 1:
      tf_unimpl(additional_msg="only sorting on last dimension is supported "
                               "for XlaSort")
    if len(args) > 2:
      tf_unimpl(additional_msg=(
        "sorting more than 2 arrays is not supported for XlaSort"))

  if prim is lax.population_count_p:
    if np_dtype in [np.uint32, np.uint64]:
      tf_unimpl(np_dtype)

  if prim is lax.clamp_p:
    if np_dtype in [np.int8, np.uint16, np.uint32, np.uint64]:
      tf_unimpl(np_dtype)

  # Testing with matmul (TODO: comment out and test without matmul)
  if prim is lax.dot_general_p:
    np_dtype = _to_np_dtype(args[0].dtype)
    if np_dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8]:
      tf_unimpl(np_dtype)
    elif np_dtype == np.int16:
      # TODO(bchetioui): the path using 'einsum' is not compatible with int16
      # arguments on CPU/GPU, while the one using 'matmul' is (but not in
      # compiled mode).
      tf_unimpl(np_dtype, additional_msg=("only cases representable as 2D "
                                          "matrix multiplication can be "
                                          "converted properly"),
                devs=['CPU', 'GPU'])
      tf_unimpl(np_dtype, devs=['TPU'])
    elif np_dtype in [np.int16, np.int64]:
      devs = ['CPU'] if np_dtype == np.int16 else ['CPU', 'GPU']
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"),
                devs=devs)
  if prim is lax.conv_general_dilated_p:
    batch_group_count = kwargs['batch_group_count']
    if batch_group_count != 1:
      tf_unimpl(additional_msg="batch_group_count != 1 unsupported")
    if np_dtype in [np.complex64, np.complex128]:
      tf_unimpl(np_dtype, additional_msg="likely bug in the HLO -> LLVM IR "
                                         "lowering of XlaConv")

  if prim in [lax.acosh_p, lax.asinh_p, lax.atanh_p, lax.bessel_i0e_p,
              lax.bessel_i1e_p, lax.digamma_p, lax.erf_p, lax.erf_inv_p,
              lax.erfc_p, lax.lgamma_p, lax.round_p, lax.rsqrt_p]:
    if np_dtype == dtypes.bfloat16:
      tf_unimpl(np_dtype, devs=["CPU", "GPU"])

  if prim in [lax.sinh_p, lax.cosh_p, lax.atanh_p, lax.asinh_p, lax.acosh_p,
              lax.erf_inv_p]:
    if np_dtype == np.float16:
      # b/158006398: float16 support missing from the kernel of the above
      # operations.
      tf_unimpl(np_dtype)

  if prim in [lax.le_p, lax.lt_p, lax.ge_p, lax.gt_p]:
    if np_dtype in [np.bool_, np.uint16, np.uint32, np.uint64]:
      tf_unimpl(np_dtype)

  if prim is lax.fft_p:
    if np_dtype in [np.float64, np.complex128]:
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"))

  if prim is lax.top_k_p:
    if np_dtype in [np.float64, np.int64, np.uint64]:
      tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
                                          "mode (experimental_compile=True))"))
  return limitations

def merge_similar_limitations(limitations: Sequence[Limitation]) \
    -> Sequence[Limitation]:
  """Merges similar limitations together."""
  merged_limitations: Dict[Limitation, Set[NpDType]] = defaultdict(set)

  for lim in limitations:
    key = Limitation(lim.primitive_name, lim.error_type, lim.error_string,
                     tuple(), lim.devices)
    value = lim.affected_dtypes
    merged_limitations[key].update(value)

  def _perform_merge(key: Limitation, values: Set[NpDType]) -> Limitation:
    return key._replace(affected_dtypes=tuple(values))

  return list(map(lambda kvp: _perform_merge(*kvp), merged_limitations.items()))

def prettify(limitations: Sequence[Limitation]) -> str:
  """Constructs a summary markdown table based on a list of limitations."""
  limitations = list(set(limitations))

  def _pipewrap(columns):
    return '| ' + ' | '.join(columns) + ' |'

  column_names = [ 'Affected primitive'
                 , 'Type of limitation'
                 , 'Description'
                 , 'Affected dtypes'
                 , 'Affected devices' ]

  header = [column_names, ['---'] * len(column_names)]
  table = []

  for lim in limitations:
    table.append([ lim.primitive_name
                 , lim.error_type
                 , lim.error_string
                 , ('ALL' if len(lim.affected_dtypes) == 0 else
                    ', '.join(sorted(list(map(str, lim.affected_dtypes)))))
                 , ', '.join(lim.devices)
                 ])

  return '\n'.join(line for line in map(_pipewrap, header + sorted(table)))

def prettify_as_ordered_list(collec: Collection[core.Primitive]) -> str:
  """Builds an ordered summary markdown list of a collection of primitives."""
  ordered_list: List[str] = sorted(list(map(lambda prim: prim.name, collec)))

  backtick_wrap = lambda prim_name: f'`{prim_name}`'
  return ', '.join(list(map(backtick_wrap, ordered_list)))

prettify_not_yet_implemented = lambda: prettify_as_ordered_list(tf_not_yet_impl)

def prettify_not_yet_covered(covered_set: Set[core.Primitive]) -> str:
  """
  Builds an ordered summary markdown list of all the primitives that are
  implemented but not in the set passed as an argument.
  """
  ignore = set(xla.call_translations)
  not_yet_covered = (
    set(filter(lambda prim: not prim in ignore, set(tf_impl) - covered_set)))

  return prettify_as_ordered_list(not_yet_covered)

def pprint_limitations(limitations: Sequence[Limitation],
                       covered_primitives: Set[core.Primitive],
                       output_file: str, template_file: str) -> None:

  limitations = merge_similar_limitations(limitations)
  limited_support_table = prettify(limitations)
  not_yet_impl_primitives = prettify_not_yet_implemented()
  not_yet_covered_primitives = prettify_not_yet_covered(covered_primitives)

  generation_date = str(datetime.date.today())

  with open(template_file, 'r') as f:
    output = f.read()

  output = (output
    .replace('{{limited-support-table}}', limited_support_table)
    .replace('{{generation-date}}', generation_date)
    .replace('{{not-yet-impl-primitives}}', not_yet_impl_primitives)
    .replace('{{not-yet-covered-primitives}}', not_yet_covered_primitives))

  with open(output_file, 'w') as f:
    f.write(output)

all_limitations: Sequence[Limitation] = []
covered_primitives: Set[core.Primitive] = set()

pprint_all_limitations = functools.partial(pprint_limitations, all_limitations,
                                           covered_primitives)

def collect_limitations(prim: core.Primitive, func: Callable) -> Callable:
  """
  Wraps a primitive and its corresponding TF implementation with `categorize`.
  """
  def wrapper(*args, **kwargs):
    global all_limitations, covered_primitives
    covered_primitives.add(prim)
    all_limitations += categorize(prim, *args, **kwargs)
    return func(*args, **kwargs)
  return wrapper
