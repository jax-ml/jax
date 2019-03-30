from jax.interpreters import polysimp
from jax import lax

import jax.linear_util as lu

polysimp.addition_primitives.add(lax.add_p)
polysimp.multiplication_primitives.add(lax.mul_p)
polysimp.linear_primitives.add(lax.broadcast_p)

f = lu.wrap_init(lambda x: x + x * x + 3 * x + 4 * x * x)
print polysimp.polysimp(f).call_wrapped((1,))
