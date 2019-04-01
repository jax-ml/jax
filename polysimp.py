from jax.interpreters import polysimp
from jax import lax
from jax import make_jaxpr
import jax.numpy as np

import jax.linear_util as lu

polysimp.addition_primitives.add(lax.add_p)
polysimp.multiplication_primitives.add(lax.mul_p)
polysimp.multiplication_primitives.add(lax.dot_p)
polysimp.linear_primitives.add(lax.broadcast_p)
polysimp.linear_primitives.add(lax.convert_element_type_p)
polysimp.linear_primitives.add(lax.reshape_p)

f = lambda x: x + x * x * x + 3 * x + 4 * x * x * x

print make_jaxpr(f)(2)
print f(2)

print make_jaxpr(polysimp.polysimp(lu.wrap_init(f)).call_wrapped)((2,))
print polysimp.polysimp(lu.wrap_init(f)).call_wrapped((2,))


import numpy as onp
rng = onp.random.RandomState(0)
A = rng.randn(2, 3)
B = rng.randn(3, 4)
f = lambda x: np.dot(A * x, B * x) + x

print make_jaxpr(f)(2)
print f(2)

print make_jaxpr(polysimp.polysimp(lu.wrap_init(f)).call_wrapped)((2,))
print polysimp.polysimp(lu.wrap_init(f)).call_wrapped((2,))
