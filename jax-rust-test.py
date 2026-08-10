import rustyjax as jax

def foo(x):
  return jax.mul(jax.add(x, 4), x)

# trace to a jaxpr
traced = jax.trace(foo, (jax.int_type,))

compiled = traced.compile()

# run interpreter
ans = compiled.eval((2,))
print(ans)  # should print `12`


