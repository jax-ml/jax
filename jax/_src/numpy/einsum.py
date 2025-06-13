# Copyright 2025 The JAX Authors.
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

import collections
from typing import overload, Any, Callable, Sequence

import numpy as np
import opt_einsum

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src.api import jit, named_call
from jax._src.export import shape_poly
from jax._src.lax import lax
from jax._src.lax.lax import PrecisionLike
from jax._src.numpy import util
from jax._src.sharding_impls import canonicalize_sharding, NamedSharding
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.util import partition_list, set_module, unzip2


export = set_module('jax.numpy')


class Unoptimized(opt_einsum.paths.PathOptimizer):
  """Unoptimized path for einsum."""
  def __call__(self, inputs, *args, **kwargs):
    return [(0, 1)] * (len(inputs) - 1)

@overload
def einsum(
    subscript: str, /,
    *operands: ArrayLike,
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
    out_sharding=None,
) -> Array: ...

@overload
def einsum(
    arr: ArrayLike,
    axes: Sequence[Any], /,
    *operands: ArrayLike | Sequence[Any],
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
    out_sharding=None,
) -> Array: ...

@export
def einsum(
    subscripts, /,
    *operands,
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
    out_sharding=None,
) -> Array:
  """Einstein summation

  JAX implementation of :func:`numpy.einsum`.

  ``einsum`` is a powerful and generic API for computing various reductions,
  inner products, outer products, axis reorderings, and combinations thereof
  across one or more input arrays. It has a somewhat complicated overloaded API;
  the arguments below reflect the most common calling convention. The Examples
  section below demonstrates some of the alternative calling conventions.

  Args:
    subscripts: string containing axes names separated by commas.
    *operands: sequence of one or more arrays corresponding to the subscripts.
    optimize: specify how to optimize the order of computation. In JAX this defaults
      to ``"auto"`` which produces optimized expressions via the opt_einsum_
      package. Other options are ``True`` (same as ``"optimal"``), ``False``
      (unoptimized), or any string supported by ``opt_einsum``, which
      includes ``"optimal"``, ``"greedy"``, ``"eager"``, and others. It may also
      be a pre-computed path (see :func:`~jax.numpy.einsum_path`).
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.
    out: unsupported by JAX
    _dot_general: optionally override the ``dot_general`` callable used by ``einsum``.
      This parameter is experimental, and may be removed without warning at any time.

  Returns:
    array containing the result of the einstein summation.

  See also:
    :func:`jax.numpy.einsum_path`

  Examples:
    The mechanics of ``einsum`` are perhaps best demonstrated by example. Here we
    show how to use ``einsum`` to compute a number of quantities from one or more
    arrays. For more discussion and examples of ``einsum``, see the documentation
    of :func:`numpy.einsum`.

    >>> M = jnp.arange(16).reshape(4, 4)
    >>> x = jnp.arange(4)
    >>> y = jnp.array([5, 4, 3, 2])

    **Vector product**

    >>> jnp.einsum('i,i', x, y)
    Array(16, dtype=int32)
    >>> jnp.vecdot(x, y)
    Array(16, dtype=int32)

    Here are some alternative ``einsum`` calling conventions to compute the same
    result:

    >>> jnp.einsum('i,i->', x, y)  # explicit form
    Array(16, dtype=int32)
    >>> jnp.einsum(x, (0,), y, (0,))  # implicit form via indices
    Array(16, dtype=int32)
    >>> jnp.einsum(x, (0,), y, (0,), ())  # explicit form via indices
    Array(16, dtype=int32)

    **Matrix product**

    >>> jnp.einsum('ij,j->i', M, x)  # explicit form
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.matmul(M, x)
    Array([14, 38, 62, 86], dtype=int32)

    Here are some alternative ``einsum`` calling conventions to compute the same
    result:

    >>> jnp.einsum('ij,j', M, x) # implicit form
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.einsum(M, (0, 1), x, (1,), (0,)) # explicit form via indices
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.einsum(M, (0, 1), x, (1,))  # implicit form via indices
    Array([14, 38, 62, 86], dtype=int32)

    **Outer product**

    >>> jnp.einsum("i,j->ij", x, y)
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.outer(x, y)
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)

    Some other ways of computing outer products:

    >>> jnp.einsum("i,j", x, y)  # implicit form
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.einsum(x, (0,), y, (1,), (0, 1))  # explicit form via indices
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.einsum(x, (0,), y, (1,))  # implicit form via indices
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)

    **1D array sum**

    >>> jnp.einsum("i->", x)  # requires explicit form
    Array(6, dtype=int32)
    >>> jnp.einsum(x, (0,), ())  # explicit form via indices
    Array(6, dtype=int32)
    >>> jnp.sum(x)
    Array(6, dtype=int32)

    **Sum along an axis**

    >>> jnp.einsum("...j->...", M)  # requires explicit form
    Array([ 6, 22, 38, 54], dtype=int32)
    >>> jnp.einsum(M, (..., 0), (...,))  # explicit form via indices
    Array([ 6, 22, 38, 54], dtype=int32)
    >>> M.sum(-1)
    Array([ 6, 22, 38, 54], dtype=int32)

    **Matrix transpose**

    >>> y = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.einsum("ij->ji", y)  # explicit form
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum("ji", y)  # implicit form
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum(y, (1, 0))  # implicit form via indices
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum(y, (0, 1), (1, 0))  # explicit form via indices
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.transpose(y)
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

    **Matrix diagonal**

    >>> jnp.einsum("ii->i", M)
    Array([ 0,  5, 10, 15], dtype=int32)
    >>> jnp.diagonal(M)
    Array([ 0,  5, 10, 15], dtype=int32)

    **Matrix trace**

    >>> jnp.einsum("ii", M)
    Array(30, dtype=int32)
    >>> jnp.trace(M)
    Array(30, dtype=int32)

    **Tensor products**

    >>> x = jnp.arange(30).reshape(2, 3, 5)
    >>> y = jnp.arange(60).reshape(3, 4, 5)
    >>> jnp.einsum('ijk,jlk->il', x, y)  # explicit form
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.tensordot(x, y, axes=[(1, 2), (0, 2)])
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum('ijk,jlk', x, y)  # implicit form
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum(x, (0, 1, 2), y, (1, 3, 2), (0, 3))  # explicit form via indices
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum(x, (0, 1, 2), y, (1, 3, 2))  # implicit form via indices
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)

    **Chained dot products**

    >>> w = jnp.arange(5, 9).reshape(2, 2)
    >>> x = jnp.arange(6).reshape(2, 3)
    >>> y = jnp.arange(-2, 4).reshape(3, 2)
    >>> z = jnp.array([[2, 4, 6], [3, 5, 7]])
    >>> jnp.einsum('ij,jk,kl,lm->im', w, x, y, z)
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> jnp.einsum(w, (0, 1), x, (1, 2), y, (2, 3), z, (3, 4))  # implicit, via indices
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> w @ x @ y @ z  # direct chain of matmuls
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> jnp.linalg.multi_dot([w, x, y, z])
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)

  .. _opt_einsum: https://github.com/dgasmith/opt_einsum
  """
  operands = (subscripts, *operands)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
  spec = operands[0] if isinstance(operands[0], str) else None
  path_type = 'optimal' if optimize is True else Unoptimized() if optimize is False else optimize

  # Extract __jax_array__ before passing to contract_path()
  operands = tuple(op.__jax_array__() if hasattr(op, "__jax_array__") else op
                   for op in operands)

  # Allow handling of shape polymorphism
  non_constant_dim_types = {
      type(d) for op in operands if not isinstance(op, str)
      for d in np.shape(op) if not core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    contract_path = opt_einsum.contract_path
  else:
    ty = next(iter(non_constant_dim_types))
    contract_path = _poly_einsum_handlers.get(ty, _default_poly_einsum_handler)
  # using einsum_call=True here is an internal api for opt_einsum... sorry
  operands, contractions = contract_path(
        *operands, einsum_call=True, use_blas=True, optimize=path_type)

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)  # pytype: disable=attribute-error

  jit_einsum = jit(_einsum, static_argnums=(1, 2, 3, 4, 5), inline=True)
  if spec is not None:
    jit_einsum = named_call(jit_einsum, name=spec)
  operand_arrays = list(util.ensure_arraylike_tuple("einsum", operands))
  return jit_einsum(operand_arrays, contractions, precision,
                    preferred_element_type, _dot_general, out_sharding)


# Enable other modules to override einsum_contact_path.
# Indexed by the type of the non constant dimension
_poly_einsum_handlers = {}  # type: ignore

def _default_poly_einsum_handler(*operands, **kwargs):
  dummy = collections.namedtuple('dummy', ['shape', 'dtype'])
  dummies = [dummy(tuple(d if type(d) is int else 8 for d in x.shape), x.dtype)
             if hasattr(x, 'dtype') else x for x in operands]
  mapping = {id(d): i for i, d in enumerate(dummies)}
  out_dummies, contractions = opt_einsum.contract_path(*dummies, **kwargs)
  contract_operands = [operands[mapping[id(d)]] for d in out_dummies]
  return contract_operands, contractions

@overload
def einsum_path(
    subscripts: str, /,
    *operands: ArrayLike,
    optimize: bool | str | list[tuple[int, ...]] =  ...,
) -> tuple[list[tuple[int, ...]], Any]: ...

@overload
def einsum_path(
    arr: ArrayLike,
    axes: Sequence[Any], /,
    *operands: ArrayLike | Sequence[Any],
    optimize: bool | str | list[tuple[int, ...]] =  ...,
) -> tuple[list[tuple[int, ...]], Any]: ...

@export
def einsum_path(
    subscripts, /,
    *operands,
    optimize: bool | str | list[tuple[int, ...]] = 'auto'
  ) -> tuple[list[tuple[int, ...]], Any]:
  """Evaluates the optimal contraction path without evaluating the einsum.

  JAX implementation of :func:`numpy.einsum_path`. This function calls into
  the opt_einsum_ package, and makes use of its optimization routines.

  Args:
    subscripts: string containing axes names separated by commas.
    *operands: sequence of one or more arrays corresponding to the subscripts.
    optimize: specify how to optimize the order of computation. In JAX this defaults
      to ``"auto"``. Other options are ``True`` (same as ``"optimize"``), ``False``
      (unoptimized), or any string supported by ``opt_einsum``, which
      includes ``"optimize"``,, ``"greedy"``, ``"eager"``, and others.

  Returns:
    A tuple containing the path that may be passed to :func:`~jax.numpy.einsum`, and a
    printable object representing this optimal path.

  Examples:
    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> x = jax.random.randint(key1, minval=-5, maxval=5, shape=(2, 3))
    >>> y = jax.random.randint(key2, minval=-5, maxval=5, shape=(3, 100))
    >>> z = jax.random.randint(key3, minval=-5, maxval=5, shape=(100, 5))
    >>> path, path_info = jnp.einsum_path("ij,jk,kl", x, y, z, optimize="optimal")
    >>> print(path)
    [(1, 2), (0, 1)]
    >>> print(path_info)
          Complete contraction:  ij,jk,kl->il
                Naive scaling:  4
            Optimized scaling:  3
              Naive FLOP count:  9.000e+3
          Optimized FLOP count:  3.060e+3
          Theoretical speedup:  2.941e+0
          Largest intermediate:  1.500e+1 elements
        --------------------------------------------------------------------------------
        scaling        BLAS                current                             remaining
        --------------------------------------------------------------------------------
          3           GEMM              kl,jk->lj                             ij,lj->il
          3           GEMM              lj,ij->il                                il->il

    Use the computed path in :func:`~jax.numpy.einsum`:

    >>> jnp.einsum("ij,jk,kl", x, y, z, optimize=path)
    Array([[-754,  324, -142,   82,   50],
           [ 408,  -50,   87,  -29,    7]], dtype=int32)

  .. _opt_einsum: https://github.com/dgasmith/opt_einsum
  """
  if optimize is True:
    optimize = 'optimal'
  elif optimize is False:
    optimize = Unoptimized()
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

def _removechars(s, chars):
  return s.translate(str.maketrans(dict.fromkeys(chars)))


def _einsum(
    operands: list[Array],
    contractions: Sequence[tuple[tuple[int, ...], frozenset[str], str]],
    precision,
    preferred_element_type,
    _dot_general=lax.dot_general,
    out_sharding=None,
):
  out_sharding = canonicalize_sharding(out_sharding, 'einsum')
  if out_sharding is not None and not isinstance(out_sharding, NamedSharding):
    raise NotImplementedError(
        "`out_sharding` argument of `einsum` only supports NamedSharding"
        " instances. Please file a bug if this is not enough for your use case.")
  dtypes.check_user_dtype_supported(preferred_element_type, "einsum")
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(
        *operands, return_weak_type_flag=True)
  else:
    output_weak_type = False

  def sum(x, axes):
    if dtypes.result_type(x, preferred_element_type) != x.dtype:
      x = x.astype(preferred_element_type)
    return lax.reduce(x, np.array(0, x.dtype),
                      lax.add if x.dtype != bool else lax.bitwise_or, axes)

  def sum_uniques(operand, names, uniques):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum(operand, axes)
      names = _removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = lax._delta(np.dtype('bool'), operand.shape, axes)
        operand = lax.select(eye, operand, lax.full_like(operand, 0))
        if name not in keep_names:
          operand = sum(operand, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    eq = core.definitely_equal
    keep = [not eq(operand.shape[i], 1) or j == -1 or eq(other_shape[j], 1)
            for i, j in enumerate(map(other_names.find, names))]
    sqez_axes, keep_axes = partition_list(keep, list(range(operand.ndim)))
    return lax.squeeze(operand, sqez_axes), "".join(names[i] for i in keep_axes)

  for i, (operand_indices, contracted_names_set, einstr) in enumerate(contractions):
    last_contraction = i == len(contractions) - 1
    contracted_names = sorted(contracted_names_set)
    input_str, result_names = einstr.split('->')
    input_names = input_str.split(',')

    # switch on the number of operands to be processed in this loop iteration.
    # every case here sets 'operand' and 'names'.
    if len(operand_indices) == 1:
      operand = operands.pop(operand_indices[0])
      names, = input_names
      counts = collections.Counter(names)

      # sum out unique contracted indices with a single reduce-sum
      uniques = [name for name in contracted_names if counts[name] == 1]
      operand, names = sum_uniques(operand, names, uniques)

      # for every repeated index, do a contraction against an identity matrix
      operand, names = sum_repeats(operand, names, counts, result_names)

    elif len(operand_indices) == 2:
      lhs, rhs = map(operands.pop, operand_indices)
      lhs_names, rhs_names = input_names

      # handle cases where one side of a contracting or batch dimension is 1
      # but its counterpart is not.
      lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, np.shape(rhs),
                                             rhs_names)
      rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, np.shape(lhs),
                                             lhs_names)

      lhs_counts = collections.Counter(lhs_names)
      rhs_counts = collections.Counter(rhs_names)

      # sum out unique contracted indices in lhs and rhs
      lhs_uniques = [name for name in contracted_names
                     if lhs_counts[name] == 1 and rhs_counts[name] == 0]
      lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

      rhs_uniques = [name for name in contracted_names
                     if rhs_counts[name] == 1 and lhs_counts[name] == 0]
      rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

      # for every repeated index, contract against an identity matrix
      lhs, lhs_names = sum_repeats(lhs, lhs_names, lhs_counts,
                                   result_names + rhs_names)
      rhs, rhs_names = sum_repeats(rhs, rhs_names, rhs_counts,
                                   result_names + lhs_names)

      lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
      contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
      lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
      batch_names = [x for x in result_names if x in lhs_and_rhs_names]

      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert config.dynamic_shapes.value or all(
        name in lhs_names and name in rhs_names and
        lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
        for name in contracted_names), (
          "Incompatible reduction dimensions: "
          f"lhs.shape={lhs.shape} lhs_names={lhs_names} "
          f"rhs.shape={rhs.shape} rhs_names={rhs_names}")

      # contract using dot_general
      batch_names_str = ''.join(batch_names)
      lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                  for n in contracted_names)
      deleted_names = batch_names_str + ''.join(contracted_names)
      remaining_lhs_names = _removechars(lhs_names, deleted_names)
      remaining_rhs_names = _removechars(rhs_names, deleted_names)
      # Try both orders of lhs and rhs, in the hope that one of them means we
      # don't need an explicit transpose. opt_einsum likes to contract from
      # right to left, so we expect (rhs,lhs) to have the best chance of not
      # needing a transpose.
      names = batch_names_str + remaining_rhs_names + remaining_lhs_names
      if names == result_names:
        dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
        k_out_sharding = ({} if out_sharding is None else
                          {'out_sharding': out_sharding})
        operand = _dot_general(rhs, lhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type,
                               **k_out_sharding)
      else:
        names = batch_names_str + remaining_lhs_names + remaining_rhs_names
        if not last_contraction:
          dot_general_out_sharding = None
        elif out_sharding is not None and names != result_names:
          if len(result_names) > len(out_sharding.spec):
            out_sharding = out_sharding.update(spec=
                out_sharding.spec._normalized_spec_for_aval(len(result_names)))
          spec = out_sharding.spec
          inverse_spec = tuple(spec[result_names.index(name)] for name in names)
          dot_general_out_sharding = NamedSharding(
              out_sharding.mesh, spec.update(partitions=inverse_spec))
        else:
          dot_general_out_sharding = out_sharding  # type: ignore
        dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
        dot_general_out_sharding = ({} if dot_general_out_sharding is None else  # type: ignore
                                    {'out_sharding': dot_general_out_sharding})
        operand = _dot_general(lhs, rhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type,
                               **dot_general_out_sharding)
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple(names.index(name) for name in result_names)
      operand = lax.transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  return lax._convert_element_type(operands[0], preferred_element_type,
                                   output_weak_type)

_poly_einsum_handlers[shape_poly._DimExpr] = shape_poly._einsum_contract_path
