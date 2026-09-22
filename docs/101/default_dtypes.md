(default-dtypes)=
# Default dtypes and the X64 flag
JAX serves a range of numerical computing practitioners, whose preferences
sometimes conflict. When it comes to default dtypes, there are two camps:

- Classic scientific computing practitioners (e.g., users of tools like {mod}`numpy` or
  {mod}`scipy`) tend to value accuracy above all, so they prefer computations to
  default to the **widest available representation**: floating point values should
  default to `float64`, integers to `int64`, and so on.
- AI researchers (e.g., people implementing and training neural networks) tend to value
  speed over accuracy, to the point that they've developed special data types like
  [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), which
  deliberately discard the least significant bits in order to speed up computation.
  For these users, the mere presence of a `float64` value in a computation can make a
  program slow at best, and incompatible with their hardware at worst. They prefer
  computations to default to `float32` or `int32`.

The main mechanism JAX offers for choosing between these is the `jax_enable_x64`
flag, which controls whether 64-bit values can be created at all. By default the
flag is `False`, serving AI researchers and practitioners, but users who value
accuracy over speed can set it to `True`.

## Default setting: 32 bits everywhere
By default `jax_enable_x64` is `False`, so {mod}`jax.numpy` array creation
functions return 32-bit values.

For example:
```python
>>> import jax.numpy as jnp

>>> jnp.arange(5)
Array([0, 1, 2, 3, 4], dtype=int32)

>>> jnp.zeros(5)
Array([0., 0., 0., 0., 0.], dtype=float32)

>>> jnp.ones(5, dtype=int)
Array([1, 1, 1, 1, 1], dtype=int32)

```

Beyond the defaults, because 64-bit values can be so harmful to AI workflows,
the flag being `False` prevents you from creating 64-bit arrays at all. Asking
for one produces a 32-bit array instead, with a warning:
```
>>> jnp.arange(5, dtype='float64')  # doctest: +SKIP
UserWarning: Explicitly requested dtype float64 requested in arange is not available, and will be 
truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the 
JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
Array([0., 1., 2., 3., 4.], dtype=float32)
```

## The X64 flag: enabling 64-bit values
To work in the other mode, where functions default to producing 64-bit values,
set the `jax_enable_x64` flag to `True`:
```python
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

print(repr(jnp.arange(5)))
print(repr(jnp.zeros(5)))
print(repr(jnp.ones(5, dtype=int)))
```
```
Array([0, 1, 2, 3, 4], dtype=int64)
Array([0., 0., 0., 0., 0.], dtype=float64)
Array([1, 1, 1, 1, 1], dtype=int64)
```

You can also set the flag with the `JAX_ENABLE_X64` shell environment variable:
```bash
$ JAX_ENABLE_X64=1 python main.py
```

The X64 flag works best as a **global setting**, with one value for your whole
program, set at the top of your main file. If you need 64-bit values in just
one section of a program, {func}`jax.enable_x64` is also available as a context
manager:
```python
with jax.enable_x64(True):
  x = jnp.arange(5)  # int64
```
Use it sparingly: arrays created inside the context keep their 64-bit dtypes
after it exits, but operations on them outside the context truncate the results
back to 32 bits.
