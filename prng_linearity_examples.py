from jax.random import PRNGKey, normal, split



###

# 1. orchestrate mode splitting

# should work
key = PRNGKey(0)
k1, k2 = split(key)
x1 = normal(k1, ())
x2 = normal(k2, ())

# should error
try:
  key = PRNGKey(0)
  for i in range(3):
    x = normal(key, ())
    _, key = split(key)
except:
  pass
else:
  raise RuntimeError


# 2. within-jit splitting

# should work
@jit
def f(i):
  key = PRNGKey(i)
  k1, k2 = split(key)
  x1 = normal(k1, ())
  x2 = normal(k2, ())
  return x1 + x2
f(0)

# should error
@jit
def f(i):
  key = PRNGKey(i)
  for i in range(3):
    x = normal(key, ())
    _, key = split(key)
  return x

try:
  f(0)
except:
  pass
else:
  raise RuntimeError

# 3. argument to jit function
key = PRNGKey(0)

@jit
def f(key):
  return normal(key, ())

f(key)  # should work
try:
  f(key)  # should error
except:
  pass
else:
  raise RuntimeError
try:
  split(key)  # should error
except:
  pass
else:
  raise RuntimeError

key2 = PRNGKey(0)
f(key2)  # should work (bonus: cache hit)


# 4. closed-over key

key = PRNGKey(0)

@jit
def f(x):
  return x + normal(key, ())

f(0.)  # should work

# should error (?) not b/c reusing key, but reusing values from the key
try:
  f(0.)
except:
  pass
else:
  raise RuntimeError

