from jax.random import PRNGKey, normal, split

# should work
key = PRNGKey(0)
k1, k2 = split(key)
x1 = normal(k1, ())
x2 = normal(k2, ())

# should error
key = PRNGKey(0)
for i in range(3):
  x = normal(key, ())
  _, key = split(key)
