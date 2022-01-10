# Copyright 2021 Google LLC
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

from collections import Counter
from dataclasses import dataclass
import itertools as it
from typing import NamedTuple, Hashable, Dict, Iterable, Tuple, Any, List

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import core
from jax._src.lax.control_flow import _scan
from jax.config import config
from jax.tree_util import tree_structure
import jax._src.test_util as jtu

from jax.experimental import gelt

config.parse_flags_with_absl()

### Example: model of PRNGKey, registered as a gelt type directly

Array = Any

class PRNGKey:
  key: Array
  def __init__(self, key):
    self.key = key
  def __repr__(self) -> str:
    return f'PRNGKey[{tuple(self.key)}]'

@dataclass(frozen=True)
class PRNGKeyTy:
  pass

def _flatten_prngkey(k: PRNGKey) -> Tuple[List[Array], PRNGKeyTy]:
  return [k.key], PRNGKeyTy()
def _unflatten_prngkey(arrays: List[Array], ty: PRNGKeyTy) -> PRNGKey:
  del ty
  k, = arrays
  return PRNGKey(k)
gelt.register_garray_elt(PRNGKey, PRNGKeyTy,
                         _flatten_prngkey, _unflatten_prngkey)

### Example: polynomials and rational functions, registered as pytrees

@jax.tree_util.register_pytree_node_class
class Monomial(dict):
  def __init__(self, degrees: Dict[Hashable, int]):
    assert all(d > 0 for d in degrees.values())
    super().__init__(degrees)

  @property
  def total_degree(self):
    return sum(self.values())

  def __hash__(self):
    return hash(frozenset(self.items()))

  def __eq__(self, other) -> bool:
    return isinstance(other, Monomial) and dict.__eq__(self, other)

  def __str__(self) -> str:
    return ' '.join(f'{key}^{exponent}' if exponent != 1 else str(key)
                    for key, exponent in sorted(self.items()))
  __repr__ = __str__

  def tree_flatten(self):
    return (), dict(self)

  @classmethod
  def tree_unflatten(cls, dct, _):
    return cls(dct)
gelt.register_garray_elt_from_pytree(Monomial)

def monomial_mul(x: Monomial, y: Monomial) -> Monomial:
  return Monomial(Counter(x) + Counter(y))
Monomial.__mul__ = monomial_mul

def monomial_key(x: Monomial, y: Monomial) -> bool:
  # assumes hashables have a total order, compares graded-lexicographically
  c = Counter({k: 0 for k in it.chain(x, y)})
  return ((-x.total_degree, *(c + Counter(x)).values()) <
          (-y.total_degree, *(c + Counter(y)).values()))

@jax.tree_util.register_pytree_node_class
class Polynomial(dict):
  def __init__(self, coeffs: Dict[Monomial, jnp.ndarray]):
    assert all(c.shape == () for c in coeffs.values())
    super().__init__(coeffs)

  def __str__(self) -> str:
    return ' + '.join(f'{c} {m}' if c != 1 or m.total_degree == 0 else str(m)
                      for m, c in self._sorted_items()).strip()
  __repr__ = __str__

  def _sorted_items(self) -> Iterable[Tuple[Monomial, jnp.ndarray]]:
    c = Counter({v:0 for m in self for v in m})
    monomial_key = lambda mon_coeff: tuple((c + Counter(mon_coeff[0])).values())
    return sorted(self.items(), key=monomial_key)

  def tree_flatten(self):
    return tuple(self._sorted_items()), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(dict(children))
gelt.register_garray_elt_from_pytree(Polynomial)

def polynomial_add(x: Polynomial, y: Polynomial) -> Polynomial:
  coeffs = dict(x)
  for mon, coeff in y.items():
    coeffs[mon] = coeffs.get(mon, 0) + coeff
  return Polynomial(coeffs)
Polynomial.__add__ = Polynomial.__radd__ = polynomial_add

def polynomial_mul(x: Polynomial, y: Polynomial) -> Polynomial:
  coeffs = {}
  for (m1, c1), (m2, c2) in it.product(x.items(), y.items()):
    mon = m1 * m2
    coeffs[mon] = coeffs.get(mon, 0) + c1 * c2
  return Polynomial(coeffs)
Polynomial.__mul__ = Polynomial.__rmul__ = polynomial_mul

class RationalFunction(NamedTuple):
  p: Polynomial
  q: Polynomial

  def __str__(self) -> str:
    p_str = str(self.p)
    q_str = str(self.q)
    l = max(len(p_str), len(q_str))
    return f'{p_str.center(l)}\n{"-" * l}\n{q_str.center(l)}'

def rational_mul(x: RationalFunction, y: RationalFunction) -> RationalFunction:
  return RationalFunction(x.p * y.p, x.q * y.q)

gelt.register_garray_elt_from_pytree(RationalFunction)


scalar_jaxpr_ty = gelt.JaxprTy(core.ShapedArray((), jnp.dtype('int32')))

class GeneralElementTypeTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(type(e).__name__), "e": e, "ety": ety}
    for e, ety in [
      # garray of jnp.ndarrays
      (jnp.array([1., 2., 3.], 'float32'),
       gelt.JaxprTy(core.ShapedArray((3,), jnp.dtype('float32')))),
      # garray of tuples
      ((1, 2, 3), gelt.PyTreeEtype('tuple', tree_structure((1, 2, 3)))),
      # garray of dicts
      ({'hi': 1, 'there': 2},
       gelt.PyTreeEtype('dict', tree_structure({'hi': 1, 'there': 2}))),
      # garray of PRNGKeys
      (PRNGKey(jnp.array([0, 1], 'uint32')), PRNGKeyTy()),
      # garray of garrays
      (gelt.Garray([jnp.arange(3, dtype='int32')], scalar_jaxpr_ty, (3,)),
       gelt.GarrayTy((3,), scalar_jaxpr_ty)),
    ]))
  def test_round_trip(self, e, ety):
    g = gelt.elt_to_rank0_garray(e)
    self.assertIsInstance(g, gelt.Garray)
    self.assertEqual(g.shape, ())
    self.assertIsInstance(g.etype, type(ety))
    self.assertEqual(g.etype, ety)
    e_ = gelt.rank0_garray_to_elt(g)
    self.assertIsInstance(e_, type(e))

  def test_jit_simple_prngkey(self):
    def make_key(seed: jnp.ndarray) -> gelt.Garray:
      assert seed.shape == (2,)
      elt = PRNGKey(seed)
      return gelt.elt_to_rank0_garray(elt)

    key = jax.jit(make_key)(jnp.array([0, 1], 'uint32'))  # doesn't crash
    self.assertIsInstance(key, gelt.Garray)
    self.assertEqual(key.shape, ())
    self.assertEqual(key.etype, PRNGKeyTy())
    self.assertLen(key.data, 1)
    self.assertEqual(key.data[0].shape, (2,))

  @parameterized.named_parameters(
    {"testcase_name":
     f"_inner_jit={inner_jit is jax.jit}_outer_jit={outer_jit is jax.jit}",
     "inner_jit": inner_jit, "outer_jit": outer_jit}
    for inner_jit, outer_jit in it.product([lambda x: x, jax.jit], repeat=2))
  def test_vmap_simple_prngkey(self, inner_jit, outer_jit):
    def make_key(seed: jnp.ndarray) -> gelt.Garray:
      assert seed.shape == (2,)
      elt = PRNGKey(seed)
      return gelt.elt_to_rank0_garray(elt)

    seeds = jnp.arange(6, dtype='uint32').reshape(3, 2)
    ks = outer_jit(jax.vmap(inner_jit(make_key)))(seeds)
    self.assertIsInstance(ks, gelt.Garray)
    self.assertEqual(ks.shape, (3,))
    self.assertEqual(ks.etype, PRNGKeyTy())
    self.assertLen(ks.data, 1)
    self.assertEqual(ks.data[0].shape, (3, 2))

  def test_vmap_simple_tuple(self):
    def make_pair(x, y) -> gelt.Garray:
      return gelt.elt_to_rank0_garray((x, y))
    ys = jax.vmap(make_pair)(jnp.array([1., 2.]), jnp.array([[1, 2, 3], [4, 5, 6]]))
    self.assertIsInstance(ys, gelt.Garray)
    self.assertEqual(ys.shape, (2,))
    treedef = jax.tree_util.tree_structure((1, 2))
    self.assertEqual(ys.etype, gelt.PyTreeEtype('tuple', treedef))
    self.assertLen(ys.data, 2)
    self.assertEqual(ys.data[0].shape, (2,))
    self.assertEqual(ys.data[1].shape, (2, 3))

    def tuple_get(x: gelt.Garray, idx: int) -> Any:
      tup = gelt.rank0_garray_to_elt(x)
      return tup[idx]
    a = jax.vmap(tuple_get, (0, None))(ys, 1)
    self.assertEqual(a.shape, (2, 3))

  def test_scan_tuple_array_input(self):
    def make_pair(x, y) -> gelt.Garray:
      return gelt.elt_to_rank0_garray((x, y))
    x1s = jnp.array([1., 2.])
    x2s = jnp.array([[1, 2, 3], [4, 5, 6]])
    array_of_pairs = jax.vmap(make_pair)(x1s, x2s)

    def scanned_fun(c, x):
      x1, x2 = gelt.rank0_garray_to_elt(x)
      return c + x1, x2

    c, ys = _scan(scanned_fun, jnp.array(0., x1s.dtype), array_of_pairs,
                     length=2)
    self.assertIsInstance(c, jnp.ndarray)
    self.assertAllClose(c, jnp.array(3., x1s.dtype), check_dtypes=True)
    self.assertIsInstance(ys, jnp.ndarray)
    self.assertAllClose(ys, x2s, check_dtypes=True)

  def test_scan_tuple_array_output(self):
    def make_pair(x, y) -> gelt.Garray:
      return gelt.elt_to_rank0_garray((x, y))
    x1 = jnp.float32(0)
    x2s = jnp.array([[1, 2, 3], [4, 5, 6]])

    def scanned_fun(c, x):
      return c, make_pair(c, x)

    _, ys = _scan(scanned_fun, x1, x2s, length=2)
    self.assertIsInstance(ys, gelt.Garray)
    treedef = jax.tree_util.tree_structure((1, 2))
    self.assertEqual(ys.etype, gelt.PyTreeEtype('tuple', treedef))

    def tuple_get(x: gelt.Garray, idx: int) -> Any:
      tup = gelt.rank0_garray_to_elt(x)
      return tup[idx]
    a = jax.vmap(tuple_get, (0, None))(ys, 0)
    b = jax.vmap(tuple_get, (0, None))(ys, 1)
    self.assertAllClose(a, jnp.zeros(2, 'float32'), check_dtypes=True)
    self.assertAllClose(b, x2s, check_dtypes=True)

  def test_polynomial_binops(self):
    def make_poly(indet: str) -> gelt.Garray:
      p = Polynomial({Monomial({indet: 1}): jnp.array(1.)})
      return gelt.elt_to_rank0_garray(p)

    def make_const(const: jnp.ndarray) -> gelt.Garray:
      assert const.shape == ()
      p = Polynomial({Monomial({}): const})
      return gelt.elt_to_rank0_garray(p)

    xs = jax.vmap(make_poly, None, axis_size=5)('x')
    cs = jax.vmap(make_const)(jnp.arange(5.))
    polys = xs * xs * xs + xs * cs

    self.assertIsInstance(polys, gelt.Garray)
    self.assertLen(polys.data, 2)

    p = gelt.rank0_garray_to_elt(polys[3])
    self.assertEqual(str(p), '3.0 x + x^3')

    p = gelt.rank0_garray_to_elt(polys[2])
    self.assertEqual(str(p), '2.0 x + x^3')

    p = gelt.rank0_garray_to_elt(polys[1])
    self.assertEqual(str(p), 'x + x^3')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
