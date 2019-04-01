from jax.interpreters import polysimp
from jax import lax
from jax import make_jaxpr

import jax.linear_util as lu

polysimp.addition_primitives.add(lax.add_p)
polysimp.multiplication_primitives.add(lax.mul_p)
polysimp.multiplication_primitives.add(lax.dot_p)
polysimp.linear_primitives.add(lax.broadcast_p)

# 4x + 5x**2
f = lambda x: x + x * x + 3 * x + 4 * x * x

print make_jaxpr(f)(2)
print f(2)

print make_jaxpr(polysimp.polysimp(lu.wrap_init(f)).call_wrapped)((2,))
print polysimp.polysimp(lu.wrap_init(f)).call_wrapped((2,))
