from jax import core
from jax import linear_util as lu
from jax.util import partial
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
import jax.numpy as np
from jax import lax

from jax import jvp, jit

from jax.config import config
config.update('jax_device_values', False)

primitive = lambda f: partial(core.call, lu.wrap_init(f), __jax_no_free_vars=True)

def _logaddexp(x, y):
  a = np.maximum(x, y)
  return a + np.log(np.add(np.exp(x - a), np.exp(y - a)))
logaddexp = primitive(_logaddexp)  # TODO make @primitive decorator

print logaddexp(2., 3.)
print jvp(logaddexp, (2., 3.), (0.5, 1.5))
ad.composite_jvps[_logaddexp] = lambda primals, tangents: (primals, tangents)
print jvp(logaddexp, (2., 3.), (0.5, 1.5))

####

def defvjp(prim, vjp):
  dummy_jvp_p = core.Primitive('dummy')
  ad.primitive_jvps[prim] = partial(jvp_with_dummy, prim, dummy_jvp_p)

  def dummy_transpose_rule(g, t, ans, *primals):
    assert t is None
    return [vjp(g, ans, *primals), None] + [None] * len(primals)

  ad.primitive_transposes[dummy_jvp_p] = dummy_transpose_rule
  dummy_jvp_p.def_abstract_eval(lambda _, ans, *primals: prim.abstract_eval(*primals))

def jvp_with_dummy(prim, dummy_jvp_p, primals, tangents):
  ans = prim.bind(*primals)
  tangents = map(ad.instantiate_zeros, primals, tangents)
  out_tangents = dummy_jvp_p.bind(core.pack(tangents), ans, *primals)
  return ans, out_tangents

defvjp(lax.mul_p, lambda g, ans, x, y: [g * y, x * g])

from jax import grad
print grad(lax.mul)(2., 3.)

####

def abstract_fun(fun, *avals):
  pvs_in = [pe.PartialVal((a, core.unit)) for a in avals]
  _, pvout, _ = pe.trace_unwrapped_to_jaxpr(fun, pvs_in)
  aval_out, _ = pvout
  return aval_out

def primitive(fun):
  fun_p = core.Primitive('custom_primitive')  # TODO get name
  fun_p.def_impl(fun)

  fun_p.def_abstract_eval(partial(abstract_fun, fun))
  ad.primitive_jvps[fun_p] = partial(jvp, fun)
  # TODO iterate through transformations, set up generic rules

  return fun_p

def logsumexp_raw(x):
  max_x = np.max(x)
  return max_x + np.log(np.sum(np.exp(x - max_x)))
logsumexp_p = primitive(logsumexp_raw)
logsumexp = lambda x: logsumexp_p.bind(x)

x = np.array([2., 3.])
t = np.array([4., 5.])
print jvp(logsumexp, (x,), (t,))
print jit(logsumexp)(x)

# def logsumexp_vjp(g, ans, x):
#   return [np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))]
# defvjp(logsumexp_p, logsumexp_vjp)

# from jax import grad
# def example_func(y):
#   z = y**2
#   lse = logsumexp(z)
#   return np.sum(lse)
# grad_of_example = grad(example_func)
# print "Gradient: ", grad_of_example(np.array([1.5, 6.7, 1e-10]))
