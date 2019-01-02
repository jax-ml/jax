from jax import core
from jax import jit
from jax.interpreters import xla, ad


# model of most jax primitives

import numpy as onp

# raw numpy for implementation (could be fortran)
def logaddexp_raw(x, y):
  a = onp.maximum(x, y)
  return a + onp.log(onp.add(onp.exp(x - a), onp.exp(y - a)))

# create 'primitive' object, associate 'impl' with raw function
logaddexp_p = core.Primitive('logaddexp')
logaddexp_p.def_impl(logaddexp_raw)

# set up the traceable
def logaddexp(x, y):
  return logaddexp_p.bind(x, y)


# at this point, we can evaluate programs but not transform them
def foo(x):
  return logaddexp(x, x)
print foo(3.)


jit_foo = jit(foo)
# print jit_foo(3.)  # error!

# two things to do:
#  1. add an abstract eval rule (for shape propagation / partiale evaluation)
#  2. add a translation rule

logaddexp_p.def_abstract_eval(lambda abstract_x, abstract_y: abstract_x)  # TODO join
# print jit_foo(3.)  # still error, but different one!


def logaddexp_translation_rule(c, xla_x, xla_y):
  xla_a = c.Max(xla_x, xla_y)
  return c.Add(xla_a, c.Log(c.Add(c.Exp(c.Sub(xla_x, xla_a)),
                                  c.Exp(c.Sub(xla_y, xla_a)))))

xla.translations[logaddexp_p] = logaddexp_translation_rule
print jit_foo(3.)


# trace-specific primitives

# let's try that another way!

def logaddexp(x, y):
  return named_call('logaddexp', x, y)


# defining a custom vjp by short-circuiting linearize-transpose

