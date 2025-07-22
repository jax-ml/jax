import timeline
import jax
import jax.numpy as jnp


def g(a, b):
    return a + b

@jax.jit
def f(a, b):
    with timeline.Timeline() as t:
        future = t.async_call(g)(a, b)
        return t.ready(future)


print(f.lower(1.0, 2.0).compile().as_text())

