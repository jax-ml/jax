# Copyright 2022 Google LLC
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
from absl.testing import absltest
import jax
from jax import core
from jax import lax
from jax import linear_util as lu
from jax.config import config
from jax.interpreters import partial_eval as pe
from jax._src import test_util as jtu
import jax.numpy as jnp

from jax._src import state

config.parse_flags_with_absl()

class StatePrimitivesTest(jtu.JaxTestCase):

  def test_cant_eval_get_primitive(self):
    with self.assertRaises(ValueError):
      state.get_p.bind(jnp.ones(5))

  def test_cant_eval_swap_primitive(self):
    with self.assertRaises(ValueError):
      state.swap_p.bind(jnp.ones(5), jnp.zeros(5))

  def test_cant_eval_addupdate_primitive(self):
    with self.assertRaises(ValueError):
      state.addupdate_p.bind(jnp.ones(5), jnp.zeros(5))

  def test_get_abstract_eval(self):
    ref_aval = state.ShapedArrayRef((1, 2, 3), jnp.float32)
    out_aval, effect = state.get_p.abstract_eval(ref_aval, 0)
    self.assertSetEqual(effect, {state.StateEffect})
    self.assertTupleEqual(out_aval.shape, (2, 3))
    self.assertEqual(out_aval.dtype, jnp.float32)

  def test_get_abstract_aval_must_take_in_refs(self):
    with self.assertRaises(ValueError):
      state.get_p.abstract_eval(core.ShapedArray((1, 2, 3), jnp.float32))

  def test_swap_abstract_eval(self):
    ref_aval = state.ShapedArrayRef((1, 2, 3), jnp.float32)
    val_aval = core.ShapedArray((2, 3), jnp.float32)
    out_aval, effect = state.swap_p.abstract_eval(ref_aval, val_aval, 0)
    self.assertSetEqual(effect, {state.StateEffect})
    self.assertTupleEqual(out_aval.shape, (2, 3))
    self.assertEqual(out_aval.dtype, jnp.float32)

  def test_swap_abstract_eval_must_take_in_refs(self):
    with self.assertRaises(ValueError):
      state.swap_p.abstract_eval(core.ShapedArray((1, 2, 3), jnp.float32),
                                    core.ShapedArray((1, 2, 3), jnp.float32))

  def test_swap_checks_for_correct_shapes(self):
    with self.assertRaises(ValueError):
      state.swap_p.abstract_eval(
          state.ShapedArrayRef((1, 2, 3), jnp.float32),
          core.ShapedArray((2, 3), jnp.float32))
    with self.assertRaises(ValueError):
      state.swap_p.abstract_eval(
          state.ShapedArrayRef((1, 2, 3), jnp.float32),
          core.ShapedArray((1, 2, 3, 4), jnp.float32))
    state.swap_p.abstract_eval(
        state.ShapedArrayRef((1, 2, 3), jnp.float32),
        core.ShapedArray((2, 3), jnp.float32), 1)

  def test_addupdate_abstract_eval(self):
    ref_aval = state.ShapedArrayRef((1, 2, 3), jnp.float32)
    val_aval = core.ShapedArray((2, 3), jnp.float32)
    out_avals, effect = state.addupdate_p.abstract_eval(ref_aval, val_aval,
                                                           0)
    self.assertSetEqual(effect, {state.StateEffect})
    self.assertListEqual(out_avals, [])

  def test_addupdate_abstract_eval_must_take_in_refs(self):
    with self.assertRaises(ValueError):
      state.addupdate_p.abstract_eval(core.ShapedArray((1, 2, 3), jnp.float32),
                                    core.ShapedArray((1, 2, 3), jnp.float32))

  def test_addupdate_checks_for_correct_shapes(self):
    with self.assertRaises(ValueError):
      state.addupdate_p.abstract_eval(
          state.ShapedArrayRef((1, 2, 3), jnp.float32),
          core.ShapedArray((2, 3), jnp.float32))
    with self.assertRaises(ValueError):
      state.addupdate_p.abstract_eval(
          state.ShapedArrayRef((1, 2, 3), jnp.float32),
          core.ShapedArray((1, 2, 3, 4), jnp.float32))
    state.addupdate_p.abstract_eval(
        state.ShapedArrayRef((1, 2, 3), jnp.float32),
        core.ShapedArray((2, 3), jnp.float32), 1)

  def test_can_represent_get_and_swap_in_jaxprs(self):

    def body(x):
      x[()] = jnp.int32(1)
      x[()] = jnp.int32(2)
      return (x[()],)
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[2].primitive, state.get_p)

  def test_can_represent_addupdate_in_jaxprs(self):

    def body(x):
      state.ref_addupdate(x, (), jnp.int32(1))
      return (x[()],)
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, state.addupdate_p)

  def test_get_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = x_ref[()]
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("b:i32[] <- a[]", jaxpr.pretty_print(use_color=False))

  def test_set_custom_pretty_printing_rule(self):
    def body(x_ref):
      x_ref[()] = jnp.int32(2)
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("a[] <- 2", jaxpr.pretty_print(use_color=False))

  def test_swap_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = state.ref_swap(x_ref, (), jnp.int32(2))
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("b:i32[], a[] <- a[], 2", jaxpr.pretty_print(use_color=False))

  def test_addupdate_custom_pretty_printing_rule(self):
    def body(x_ref):
      state.ref_addupdate(x_ref, (), jnp.int32(2))
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("a[] += 2", jaxpr.pretty_print(use_color=False))

  def test_get_jvp(self):

    def f(r):
      x = r[()]
      return jnp.cos(x)

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.get_p)

  def test_swap_jvp(self):

    def f(a):
      x = a[()]
      a[()] = jnp.sin(x)
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[2].primitive, lax.sin_p)
    self.assertEqual(jaxpr.eqns[3].primitive, lax.cos_p)
    self.assertEqual(jaxpr.eqns[4].primitive, lax.mul_p)
    self.assertEqual(jaxpr.eqns[5].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[6].primitive, state.swap_p)

  def test_addupdate_jvp(self):

    def f(a):
      state.ref_addupdate(a, (), 1.)
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.addupdate_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.addupdate_p)
    self.assertEqual(jaxpr.eqns[2].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[3].primitive, state.get_p)

class StateDischargeTest(jtu.JaxTestCase):

  def test_discharge_get(self):
    def f(a_ref):
      a = state.ref_get(a_ref, ())
      return [a + 1]
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertEqual(discharged_jaxpr.eqns[0].primitive, lax.add_p)
    # Should be able to evaluate this jaxpr
    self.assertListEqual(core.eval_jaxpr(discharged_jaxpr, (),
                                         jnp.float32(1.)), [2., 1.])

  def test_discharge_get_with_slice(self):
    def f(a_ref):
      a = state.ref_get(a_ref, (0, 1))
      return [a + 1]
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    outval, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((outval == inval[0, 1] + 1).all())
    self.assertTrue((refval == inval).all())

  def test_discharge_set(self):
    def f(a_ref, b):
      state.ref_set(a_ref, (), b + 1)
      return []
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that ignores the first
    # value and returns second value plus 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 2.)

  def test_discharge_set_with_slice(self):
    def f(a_ref):
      state.ref_set(a_ref, (0, 1), jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].set(1.)).all())

  def test_discharge_addupdate(self):
    def f(a_ref, b):
      state.ref_addupdate(a_ref, (), b + 1)
      return []
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that adds the first value,
    # second value, and 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 4.)

  def test_discharge_addupdate_with_slice(self):
    def f(a_ref):
      state.ref_addupdate(a_ref, (0, 1),
                             jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.add_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].add(1.)).all())

  def test_discharge_jaxpr_with_multiple_outputs(self):
    def f(a_ref):
      a = state.ref_get(a_ref, ())
      b = a + 1
      return [a, b]
    in_avals = [state.ShapedArrayRef((4,), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 3)
    inval = jnp.arange(4., dtype=jnp.float32)
    a, b, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((a == inval).all())
    self.assertTrue((b == inval + 1).all())
    self.assertTrue((refval == inval).all())


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
