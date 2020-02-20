from jax import jet

def f(x, y):
  return x + 2 * y

out = jet(f, (1., 2.), [(1., 0.), (1., 0.)])
print(out)
