import rustyjax as jax

def foo(x):
  return jax.mul(jax.add(x, 4), x)

# trace to a jaxpr
traced = jax.trace(foo, (jax.int_type,))

# run interpreter
ans = jax.eval(traced, (2,))
print(ans)  # should print `12`


