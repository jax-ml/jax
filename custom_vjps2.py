from functools import partial

from jax import core
from jax import jit
from jax import lax
from jax.interpreters import xla
from jax.interpreters import partial_eval as pe

import jax.numpy as np

abstractify = lax._abstractify

# let's make a named call

def _logaddexp(x, y):
  a = np.maximum(x, y)
  return a + np.log(np.add(np.exp(x - a), np.exp(y - a)))

def logaddexp(x, y):
  return named_call(_logaddexp, x, y)


def named_call(fun, *args, **kwargs):
  pvals = map(abstractify, args)  # TODO maybe don't increase abstractification
  jaxpr, pv_out, consts = pe.trace_unwrapped_to_jaxpr(partial(fun, **kwargs), pvals)
  return named_call_p.bind(*args, fun=fun,
                           jaxpr=jaxpr, pv_out=pv_out, consts=consts)

named_call_p = core.Primitive('named_call')

def named_call_translation_rule(c, *args, **kwargs):
  fun = kwargs.pop('fun')
  if fun in named_call_translations:
    return named_call_translations[fun](c, *args)
  else:
    jaxpr = kwargs.pop('jaxpr')
    abs_out, out_const = kwargs.pop('pv_out')
    consts = kwargs.pop('consts')
    assert not kwargs  # TODO fix up kwargs handling of named_call_p
    assert out_const is core.unit  # TODO generalize this to allow some consts
    in_shapes = map(c.GetShape, args)
    xla_computation = xla.jaxpr_computation(jaxpr, consts, (), *in_shapes)
    return c.Call(xla_computation, args)

named_call_p.def_impl(lambda *args, **kwargs: kwargs['fun'](*args))
named_call_p.def_abstract_eval(lambda *args, **kwargs: kwargs['pv_out'][0])

xla.translations[named_call_p] = named_call_translation_rule

named_call_translations = {}


print logaddexp(2., 3.)
print jit(logaddexp)(2., 3.)

# def custom_translation(c, xla_x, xla_y):
#   print 'here!'
#   xla_a = c.Max(xla_x, xla_y)
#   return c.Add(xla_a, c.Log(c.Add(c.Exp(c.Sub(xla_x, xla_a)),
#                                   c.Exp(c.Sub(xla_y, xla_a)))))
# named_call_translations[_logaddexp] = custom_translation
# print jit(logaddexp)(2., 3.)  # must clear caches for 'here!' to print


def named_call_jvp_rule(primals, tangents, jaxpr, pv_out, consts):
  fun = kwargs.pop('fun')
  if fun in named_call_jvps:
    return named_call_jvps[fun](primals, tangents)
  else:
    jaxpr = kwargs.pop('jaxpr')
    consts = kwargs.pop('consts')
    py_fun = lambda *args: core.eval_jaxpr(jaxpr, consts, (), *args)
    return ad.jvp(py_fun, primals, tangents)
