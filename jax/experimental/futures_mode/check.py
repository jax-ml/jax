import timeline
import jax
import jax.numpy as jnp


def g(a, b):
    return a + b

@jax.jit
def f(a, b):
    with timeline.Timeline() as t:
        future1 = t.async_call(g)(a, b)
        res = t.ready(future1)
        future2 = t.async_call(g)(b, a)
        return res + t.ready(future2)

print(f.lower(1.0, 2.0).as_text())
print(f.lower(1.0, 2.0).compile().as_text())
print(f(1.0, 2.0))

