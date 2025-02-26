---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "hjM_sV_AepYf"}

# 🔪 JAX - The Sharp Bits 🔪

<!--* freshness: { reviewed: '2024-06-03' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb)

+++ {"id": "4k5PVzEo2uJO"}

When walking about the countryside of Italy, the people will not hesitate to tell you that __JAX__ has [_"una anima di pura programmazione funzionale"_](https://www.sscardapane.it/iaml-backup/jax-intro/).

__JAX__ is a language for __expressing__ and __composing__ __transformations__ of numerical programs. __JAX__ is also able to __compile__ numerical programs for CPU or accelerators (GPU/TPU).
JAX works great for many numerical and scientific programs, but __only if they are written with certain constraints__ that we describe below.

```{code-cell} ipython3
:id: GoK_PCxPeYcy

import numpy as np
from jax import jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
```

+++ {"id": "gX8CZU1g2agP"}

## 🔪 Pure functions

+++ {"id": "2oHigBkW2dPT"}

JAX transformation and compilation are designed to work only on Python functions that are functionally pure: all the input data is passed through the function parameters, all the results are output through the function results. A pure function will always return the same result if invoked with the same inputs.

Here are some examples of functions that are not functionally pure for which JAX behaves differently than the Python interpreter. Note that these behaviors are not guaranteed by the JAX system; the proper way to use JAX is to use it only on functionally pure Python functions.

```{code-cell} ipython3
:id: A6R-pdcm4u3v
:outputId: 25dcb191-14d4-4620-bcb2-00492d2f24e1

def impure_print_side_effect(x):
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
```

```{code-cell} ipython3
:id: -N8GhitI2bhD
:outputId: fd3624c9-197d-42cb-d97f-c5e0ef885467

g = 0.
def impure_uses_globals(x):
  return x + g

# JAX captures the value of the global during the first run
print ("First call: ", jit(impure_uses_globals)(4.))
g = 10.  # Update the global

# Subsequent runs may silently use the cached value of the globals
print ("Second call: ", jit(impure_uses_globals)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
# This will end up reading the latest value of the global
print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))
```

```{code-cell} ipython3
:id: RTB6iFgu4DL6
:outputId: 16697bcd-3623-49b1-aabb-c54614aeadea

g = 0.
def impure_saves_global(x):
  global g
  g = x
  return x

# JAX runs once the transformed function with special Traced values for arguments
print ("First call: ", jit(impure_saves_global)(4.))
print ("Saved global: ", g)  # Saved global has an internal JAX value
```

+++ {"id": "Mlc2pQlp6v-9"}

A Python function can be functionally pure even if it actually uses stateful objects internally, as long as it does not read or write external state:

```{code-cell} ipython3
:id: TP-Mqf_862C0
:outputId: 78d55886-54de-483c-e7c4-bafd1d2c7219

def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']

print(jit(pure_uses_internal_state)(5.))
```

+++ {"id": "cDpQ5u63Ba_H"}

It is not recommended to use iterators in any JAX function you want to `jit` or in any control-flow primitive. The reason is that an iterator is a python object which introduces state to retrieve the next element. Therefore, it is incompatible with JAX's functional programming model. In the code below, there are some examples of incorrect attempts to use iterators with JAX. Most of them return an error, but some give unexpected results.

```{code-cell} ipython3
:id: w99WXa6bBa_H
:outputId: 52d885fd-0239-4a08-f5ce-0c38cc008903

import jax.numpy as jnp
from jax import make_jaxpr

# lax.fori_loop
array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0

# lax.scan
def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
make_jaxpr(func11)(jnp.arange(16), 5.)
# make_jaxpr(func11)(iter(range(16)), 5.) # throws error

# lax.cond
array_operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
iter_operand = iter(range(10))
# lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error
```

+++ {"id": "oBdKtkVW8Lha"}

## 🔪 In-place updates

+++ {"id": "JffAqnEW4JEb"}

In Numpy you're used to doing this:

```{code-cell} ipython3
:id: om4xV7_84N9j
:outputId: 88b0074a-4440-41f6-caa7-031ac2d1a96f

numpy_array = np.zeros((3,3), dtype=np.float32)
print("original array:")
print(numpy_array)

# In place, mutating update
numpy_array[1, :] = 1.0
print("updated array:")
print(numpy_array)
```

+++ {"id": "go3L4x3w4-9p"}

If we try to update a JAX device array in-place, however, we get an __error__!  (☉_☉)

```{code-cell} ipython3
:id: iOscaa_GecEK
:outputId: 26fdb703-a476-4b7f-97ba-d28997ef750c

%xmode Minimal
```

```{code-cell} ipython3
:id: 2AxeCufq4wAp
:outputId: fa4a87ad-1a84-471a-a3c5-a1396c432c85
:tags: [raises-exception]

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

# In place update of JAX's array will yield an error!
jax_array[1, :] = 1.0
```

+++ {"id": "7mo76sS25Wco"}

Allowing mutation of variables in-place makes program analysis and transformation difficult. JAX requires that programs are pure functions.

Instead, JAX offers a _functional_ array update using the [`.at` property on JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

+++ {"id": "hfloZ1QXCS_J"}

️⚠️ inside `jit`'d code and `lax.while_loop` or `lax.fori_loop` the __size__ of slices can't be functions of argument _values_ but only functions of argument _shapes_ -- the slice start indices have no such restriction.  See the below __Control Flow__ Section for more information on this limitation.

+++ {"id": "X2Xjjvd-l8NL"}

### Array updates: `x.at[idx].set(y)`

+++ {"id": "SHLY52KQEiuX"}

For example, the update above can be written as:

```{code-cell} ipython3
:id: PBGI-HIeCP_s
:outputId: de13f19a-2066-4df1-d503-764c34585529

updated_array = jax_array.at[1, :].set(1.0)
print("updated array:\n", updated_array)
```

+++ {"id": "zUANAw9sCmgu"}

JAX's array update functions, unlike their NumPy versions, operate out-of-place. That is, the updated array is returned as a new array and the original array is not modified by the update.

```{code-cell} ipython3
:id: dbB0UmMhCe8f
:outputId: 55d46fa1-d0de-4c43-996c-f3bbc87b7175

print("original array unchanged:\n", jax_array)
```

+++ {"id": "eM6MyndXL2NY"}

However, inside __jit__-compiled code, if the __input value__ `x` of `x.at[idx].set(y)` is not reused, the compiler will optimize the array update to occur _in-place_.

+++ {"id": "7to-sF8EmC_y"}

### Array updates with other operations

+++ {"id": "ZY5l3tAdDmsJ"}

Indexed array updates are not limited simply to overwriting values. For example, we can perform indexed addition as follows:

```{code-cell} ipython3
:id: tsw2svao8FUp
:outputId: 3c62a3b1-c12d-46f0-da74-791ec4b61e0b

print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = jax_array.at[::2, 3:].add(7.)
print("new array post-addition:")
print(new_jax_array)
```

+++ {"id": "sTjJ3WuaDyqU"}

For more details on indexed array updates, see the [documentation for the `.at` property](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

+++ {"id": "oZ_jE2WAypdL"}

## 🔪 Out-of-bounds indexing

+++ {"id": "btRFwEVzypdN"}

In Numpy, you are used to errors being thrown when you index an array outside of its bounds, like this:

```{code-cell} ipython3
:id: 5_ZM-BJUypdO
:outputId: c9c41ae8-2653-4219-e6dc-09b03faa3b95
:tags: [raises-exception]

np.arange(10)[11]
```

+++ {"id": "eoXrGARWypdR"}

However, raising an error from code running on an accelerator can be difficult or impossible. Therefore, JAX must choose some non-error behavior for out of bounds indexing (akin to how invalid floating point arithmetic results in `NaN`). When the indexing operation is an array index update (e.g. `index_add` or `scatter`-like primitives), updates at out-of-bounds indices will be skipped; when the operation is an array index retrieval (e.g. NumPy indexing or `gather`-like primitives) the index is clamped to the bounds of the array since __something__ must be returned. For example, the last value of the array will be returned from this indexing operation:

```{code-cell} ipython3
:id: cusaAD0NypdR
:outputId: af1708aa-b50b-4da8-f022-7f2fa67030a8

jnp.arange(10)[11]
```

+++ {"id": "NAcXJNAcDi_v"}

If you would like finer-grained control over the behavior for out-of-bound indices, you can use the optional parameters of [`ndarray.at`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html); for example:

```{code-cell} ipython3
:id: -0-MaFddO-xy
:outputId: 746c4b2b-a90e-4ec9-de56-ed6682d451e5

jnp.arange(10.0).at[11].get()
```

```{code-cell} ipython3
:id: g5JEJtIUPBXi
:outputId: 4a0f6854-1165-47f2-e1ac-5a21fa2b8516

jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan)
```

+++ {"id": "J8uO8yevBa_M"}

Note that due to this behavior for index retrieval, functions like `jnp.nanargmin` and `jnp.nanargmax` return -1 for slices consisting of NaNs whereas Numpy would throw an error.

Note also that, as the two behaviors described above are not inverses of each other, reverse-mode automatic differentiation (which turns index updates into index retrievals and vice versa) [will not preserve the semantics of out of bounds indexing](https://github.com/jax-ml/jax/issues/5760). Thus it may be a good idea to think of out-of-bounds indexing in JAX as a case of [undefined behavior](https://en.wikipedia.org/wiki/Undefined_behavior).

+++ {"id": "LwB07Kx5sgHu"}

## 🔪 Non-array inputs: NumPy vs. JAX

NumPy is generally happy accepting Python lists or tuples as inputs to its API functions:

```{code-cell} ipython3
:id: sErQES14sjCG
:outputId: 601485ff-4cda-48c5-f76c-2789073c4591

np.sum([1, 2, 3])
```

+++ {"id": "ZJ1Wt1bTtrSA"}

JAX departs from this, generally returning a helpful error:

```{code-cell} ipython3
:id: DFEGcENSsmEc
:outputId: 08535679-6c1f-4dd9-a414-d8b59310d1ee
:tags: [raises-exception]

jnp.sum([1, 2, 3])
```

+++ {"id": "QPliLUZztxJt"}

This is a deliberate design choice, because passing lists or tuples to traced functions can lead to silent performance degradation that might otherwise be difficult to detect.

For example, consider the following permissive version of `jnp.sum` that allows list inputs:

```{code-cell} ipython3
:id: jhe-L_TwsvKd
:outputId: ab2ee183-d9ec-45cc-d6be-5009347e1bc5

def permissive_sum(x):
  return jnp.sum(jnp.array(x))

x = list(range(10))
permissive_sum(x)
```

+++ {"id": "m0XZLP7nuYdE"}

The output is what we would expect, but this hides potential performance issues under the hood. In JAX's tracing and JIT compilation model, each element in a Python list or tuple is treated as a separate JAX variable, and individually processed and pushed to device. This can be seen in the jaxpr for the ``permissive_sum`` function above:

```{code-cell} ipython3
:id: k81u6DQ7vAjQ
:outputId: 869fc3b9-feda-4aa9-d2e5-5b5107de102d

make_jaxpr(permissive_sum)(x)
```

+++ {"id": "C0_dpCfpvCts"}

Each entry of the list is handled as a separate input, resulting in a tracing & compilation overhead that grows linearly with the size of the list. To prevent surprises like this, JAX avoids implicit conversions of lists and tuples to arrays.

If you would like to pass a tuple or list to a JAX function, you can do so by first explicitly converting it to an array:

```{code-cell} ipython3
:id: nFf_DydixG8v
:outputId: e31b43b3-05f7-4300-fdd2-40e3896f6f8f

jnp.sum(jnp.array(x))
```

+++ {"id": "MUycRNh6e50W"}

## 🔪 Random numbers

JAX's pseudo-random number generation differs from Numpy's in important ways. For a quick how-to, see {ref}`key-concepts-prngs`. For more details, see the {ref}`pseudorandom-numbers` tutorial.

+++ {"id": "rg4CpMZ8c3ri"}

## 🔪 Control flow

Moved to {ref}`control-flow`.

+++ {"id": "OxLsZUyRt_kF"}

## 🔪 Dynamic shapes

+++ {"id": "1tKXcAMduDR1"}

JAX code used within transforms like `jax.jit`, `jax.vmap`, `jax.grad`, etc. requires all output arrays and intermediate arrays to have static shape: that is, the shape cannot depend on values within other arrays.

For example, if you were implementing your own version of `jnp.nansum`, you might start with something like this:

```{code-cell} ipython3
:id: 9GIwgvfLujiD

def nansum(x):
  mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
  x_without_nans = x[mask]
  return x_without_nans.sum()
```

+++ {"id": "43S7wYAiupGe"}

Outside JIT and other transforms, this works as expected:

```{code-cell} ipython3
:id: ITYoNQEZur4s
:outputId: a9a03d25-9c54-43b6-d35e-aea6c448d680

x = jnp.array([1, 2, jnp.nan, 3, 4])
print(nansum(x))
```

+++ {"id": "guup5n8xvGI-"}

If you attempt to apply `jax.jit` or another transform to this function, it will error:

```{code-cell} ipython3
:id: nms9KjQEvNTz
:outputId: d8ae982f-111d-45b6-99f8-37715e2eaab3
:tags: [raises-exception]

jax.jit(nansum)(x)
```

+++ {"id": "r2aGyHDkvauu"}

The problem is that the size of `x_without_nans` is dependent on the values within `x`, which is another way of saying its size is *dynamic*.
Often in JAX it is possible to work-around the need for dynamically-sized arrays via other means.
For example, here it is possible to use the three-argument form of  `jnp.where` to replace the NaN values with zeros, thus computing the same result while avoiding dynamic shapes:

```{code-cell} ipython3
:id: Zbuj7Dg1wnSg
:outputId: 81a5e356-cd28-4709-b307-07c6254c82de

@jax.jit
def nansum_2(x):
  mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
  return jnp.where(mask, x, 0).sum()

print(nansum_2(x))
```

+++ {"id": "uGH-jqK7wxTl"}

Similar tricks can be played in other situations where dynamically-shaped arrays occur.

+++ {"id": "DKTMw6tRZyK2"}

## 🔪 NaNs

+++ {"id": "ncS0NI4jZrwy"}

### Debugging NaNs

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by:

* setting the `JAX_DEBUG_NANS=True` environment variable;

* adding `jax.config.update("jax_debug_nans", True)` near the top of your main file;

* adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

This will cause computations to error-out immediately on production of a NaN. Switching this option on adds a nan check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@jit`. For code under an `@jit`, the output of every `@jit` function is checked and if a nan is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jit` at a time.

There could be tricky situations that arise, like nans that only occur under a `@jit` but don't get produced in de-optimized mode. In that case you'll see a warning message print out but your code will continue to execute.

If the nans are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple jaxpr interpreter that walks the sequence of primitive operations in reverse. In the example below, we started an ipython repl with the command line `env JAX_DEBUG_NANS=True ipython`, then ran this:

+++ {"id": "p6ZtDHPbBa_W"}

```
In [1]: import jax.numpy as jnp

In [2]: jnp.divide(0., 0.)
---------------------------------------------------------------------------
FloatingPointError                        Traceback (most recent call last)
<ipython-input-2-f2e2c413b437> in <module>()
----> 1 jnp.divide(0., 0.)

.../jax/jax/numpy/lax_numpy.pyc in divide(x1, x2)
    343     return floor_divide(x1, x2)
    344   else:
--> 345     return true_divide(x1, x2)
    346
    347

.../jax/jax/numpy/lax_numpy.pyc in true_divide(x1, x2)
    332   x1, x2 = _promote_shapes(x1, x2)
    333   return lax.div(lax.convert_element_type(x1, result_dtype),
--> 334                  lax.convert_element_type(x2, result_dtype))
    335
    336

.../jax/jax/lax.pyc in div(x, y)
    244 def div(x, y):
    245   r"""Elementwise division: :math:`x \over y`."""
--> 246   return div_p.bind(x, y)
    247
    248 def rem(x, y):

... stack trace ...

.../jax/jax/interpreters/xla.pyc in handle_result(device_buffer)
    103         py_val = device_buffer.to_py()
    104         if np.any(np.isnan(py_val)):
--> 105           raise FloatingPointError("invalid value")
    106         else:
    107           return Array(device_buffer, *result_shape)

FloatingPointError: invalid value
```

+++ {"id": "_NCnVt_GBa_W"}

The nan generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@jit`, as the example below shows.

+++ {"id": "pf8RF6eiBa_W"}

```
In [4]: from jax import jit

In [5]: @jit
   ...: def f(x, y):
   ...:     a = x * y
   ...:     b = (x + y) / (x - y)
   ...:     c = a + 2
   ...:     return a + b * c
   ...:

In [6]: x = jnp.array([2., 0.])

In [7]: y = jnp.array([3., 0.])

In [8]: f(x, y)
Invalid value encountered in the output of a jit function. Calling the de-optimized version.
---------------------------------------------------------------------------
FloatingPointError                        Traceback (most recent call last)
<ipython-input-8-811b7ddb3300> in <module>()
----> 1 f(x, y)

 ... stack trace ...

<ipython-input-5-619b39acbaac> in f(x, y)
      2 def f(x, y):
      3     a = x * y
----> 4     b = (x + y) / (x - y)
      5     c = a + 2
      6     return a + b * c

.../jax/jax/numpy/lax_numpy.pyc in divide(x1, x2)
    343     return floor_divide(x1, x2)
    344   else:
--> 345     return true_divide(x1, x2)
    346
    347

.../jax/jax/numpy/lax_numpy.pyc in true_divide(x1, x2)
    332   x1, x2 = _promote_shapes(x1, x2)
    333   return lax.div(lax.convert_element_type(x1, result_dtype),
--> 334                  lax.convert_element_type(x2, result_dtype))
    335
    336

.../jax/jax/lax.pyc in div(x, y)
    244 def div(x, y):
    245   r"""Elementwise division: :math:`x \over y`."""
--> 246   return div_p.bind(x, y)
    247
    248 def rem(x, y):

 ... stack trace ...
```

+++ {"id": "6ur2yArDBa_W"}

When this code sees a nan in the output of an `@jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

⚠️ You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lots of device-host round-trips and performance regressions!

⚠️ The NaN-checker doesn't work with `pmap`. To debug nans in `pmap` code, one thing to try is replacing `pmap` with `vmap`.

+++ {"id": "YTktlwTTMgFl"}

## 🔪 Double (64bit) precision

At the moment, JAX by default enforces single-precision numbers to mitigate the Numpy API's tendency to aggressively promote operands to `double`.  This is the desired behavior for many machine-learning applications, but it may catch you by surprise!

```{code-cell} ipython3
:id: CNNGtzM3NDkO
:outputId: b422bb23-a784-44dc-f8c9-57f3b6c861b8

x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
x.dtype
```

+++ {"id": "VcvqzobxNPbd"}

To use double-precision numbers, you need to set the `jax_enable_x64` configuration variable __at startup__.

There are a few ways to do this:

1. You can enable 64-bit mode by setting the environment variable `JAX_ENABLE_X64=True`.

2. You can manually set the `jax_enable_x64` configuration flag at startup:

   ```python
   # again, this only works on startup!
   import jax
   jax.config.update("jax_enable_x64", True)
   ```

3. You can parse command-line flags with `absl.app.run(main)`

   ```python
   import jax
   jax.config.config_with_absl()
   ```

4. If you want JAX to run absl parsing for you, i.e. you don't want to do `absl.app.run(main)`, you can instead use

   ```python
   import jax
   if __name__ == '__main__':
     # calls jax.config.config_with_absl() *and* runs absl parsing
     jax.config.parse_flags_with_absl()
   ```

Note that #2-#4 work for _any_ of JAX's configuration options.

We can then confirm that `x64` mode is enabled, for example:

```python
import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)
x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
x.dtype # --> dtype('float64')
```

+++ {"id": "6Cks2_gKsXaW"}

### Caveats
⚠️ XLA doesn't support 64-bit convolutions on all backends!

+++ {"id": "WAHjmL0E2XwO"}

## 🔪 Miscellaneous divergences from NumPy

While `jax.numpy` makes every attempt to replicate the behavior of numpy's API, there do exist corner cases where the behaviors differ.
Many such cases are discussed in detail in the sections above; here we list several other known places where the APIs diverge.

- For binary operations, JAX's type promotion rules differ somewhat from those used by NumPy. See [Type Promotion Semantics](https://jax.readthedocs.io/en/latest/type_promotion.html) for more details.
- When performing unsafe type casts (i.e. casts in which the target dtype cannot represent the input value), JAX's behavior may be backend dependent, and in general may diverge from NumPy's behavior. Numpy allows control over the result in these scenarios via the `casting` argument (see [`np.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html)); JAX does not provide any such configuration, instead directly inheriting the behavior of [XLA:ConvertElementType](https://www.tensorflow.org/xla/operation_semantics#convertelementtype).

  Here is an example of an unsafe cast with differing results between NumPy and JAX:
  ```python
  >>> np.arange(254.0, 258.0).astype('uint8')
  array([254, 255,   0,   1], dtype=uint8)

  >>> jnp.arange(254.0, 258.0).astype('uint8')
  Array([254, 255, 255, 255], dtype=uint8)

  ```
  This sort of mismatch would typically arise when casting extreme values from floating to integer types or vice versa.
- When operating on [subnormal](https://en.wikipedia.org/wiki/Subnormal_number)
  floating point numbers, JAX operations use flush-to-zero semantics on some
  backends. For example:
  ```python
  >>> import jax.numpy as jnp
  >>> subnormal = jnp.float32(1E-45)
  >>> subnormal  # subnormals are representable
  Array(1.e-45, dtype=float32)
  >>> subnormal + 0  # but are flushed to zero within operations
  Array(0., dtype=float32)

  ```
  The detailed operation semantics for subnormal values will generally
  vary depending on the backend.

## 🔪 Sharp bits covered in tutorials
- {ref}`control-flow` discusses how to work with the constraints that `jit` imposes on the use of Python control flow and logical operators.
- {ref}`stateful-computations` gives some advice on how to properly handle state in a JAX program, given that JAX transformations can be applied only to pure functions.

## Fin.

If something's not covered here that has caused you weeping and gnashing of teeth, please let us know and we'll extend these introductory _advisos_!
