---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "hjM_sV_AepYf"}

# 🔪 JAX - The Sharp Bits 🔪

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/master/docs/notebooks/Common_Gotchas_in_JAX.ipynb)

+++ {"id": "4k5PVzEo2uJO"}

*levskaya@ mattjj@*

When walking about the countryside of [Italy](https://iaml.it/blog/jax-intro), the people will not hesitate to tell you that __JAX__ has _"una anima di pura programmazione funzionale"_.

__JAX__ is a language for __expressing__ and __composing__ __transformations__ of numerical programs. __JAX__ is also able to __compile__ numerical programs for CPU or accelerators (GPU/TPU). 
JAX works great for many numerical and scientific programs, but __only if they are written with certain constraints__ that we describe below.

```{code-cell} ipython3
:id: GoK_PCxPeYcy

import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
```

+++ {"id": "gX8CZU1g2agP"}

## 🔪 Pure functions

+++ {"id": "2oHigBkW2dPT"}

JAX transformation and compilation are designed to work only on Python functions that are functionally pure: all the input data is passed through the function parameters, all the results are output through the function results. A pure function will always return the same result if invoked with the same inputs. 

Here are some examples of functions that are not functionally pure for which JAX behaves differently than the Python interpreter. Note that these behaviors are not guaranteed by the JAX system; the proper way to use JAX is to use it only on functionally pure Python functions.

```{code-cell} ipython3
:id: A6R-pdcm4u3v
:outputId: 389605df-a4d5-4d4b-8d74-64e9d5d39456

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
:outputId: f16ce914-1387-43b4-9b8a-1d6e3b97b11d

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
:outputId: e93d2a70-1c18-477a-d69d-d09ed556305a

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
:outputId: 78df2d95-2c6f-41c9-84a9-feda6329e75e

def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']

print(jit(pure_uses_internal_state)(5.))
```

It is not recommended to use iterators in any JAX function you want to `jit` or in any control-flow primitive. The reason is that an iterator is a python object which introduces state to retrieve the next element. Therefore, it is incompatible with JAX functional programming model. In the code below, there are some examples of incorrect attempts to use iterators with JAX. Most of them return an error, but some give unexpected results.

```{code-cell} ipython3
import jax.numpy as jnp
import jax.lax as lax
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

## 🔪 In-Place Updates

+++ {"id": "JffAqnEW4JEb"}

In Numpy you're used to doing this:

```{code-cell} ipython3
:id: om4xV7_84N9j
:outputId: 733f901e-d433-4dc8-b5bb-0c23bf2b1306

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
:id: 2AxeCufq4wAp
:outputId: d5d873db-cee0-49dc-981d-ec852347f7ca
:tags: [raises-exception]

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

# In place update of JAX's array will yield an error!
try:
  jax_array[1, :] = 1.0
except Exception as e:
  print("Exception {}".format(e))
```

+++ {"id": "7mo76sS25Wco"}

__What gives?!__  

Allowing mutation of variables in-place makes program analysis and transformation very difficult. JAX requires a pure functional expression of a numerical program.  

Instead, JAX offers the _functional_ update functions: [__index_update__](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_update.html#jax.ops.index_update), [__index_add__](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_add.html#jax.ops.index_add), [__index_min__](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_min.html#jax.ops.index_min), [__index_max__](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_max.html#jax.ops.index_max), and the [__index__](https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index.html#jax.ops.index) helper.

️⚠️ inside `jit`'d code and `lax.while_loop` or `lax.fori_loop` the __size__ of slices can't be functions of argument _values_ but only functions of argument _shapes_ -- the slice start indices have no such restriction.  See the below __Control Flow__ Section for more information on this limitation.

```{code-cell} ipython3
:id: m5lg1RYq5D9p

from jax.ops import index, index_add, index_update
```

+++ {"id": "X2Xjjvd-l8NL"}

### index_update

+++ {"id": "eM6MyndXL2NY"}

If the __input values__ of __index_update__ aren't reused, __jit__-compiled code will perform these operations _in-place_.

```{code-cell} ipython3
:id: ygUJT49b7BBk
:outputId: 1a3511c4-a480-472f-cccb-5e01620cbe99

jax_array = jnp.zeros((3, 3))
print("original array:")
print(jax_array)

new_jax_array = index_update(jax_array, index[1, :], 1.)

print("old array unchanged:")
print(jax_array)

print("new array:")
print(new_jax_array)
```

+++ {"id": "7to-sF8EmC_y"}

### index_add

+++ {"id": "iI5cLY1xMBLs"}

If the __input values__ of __index_update__ aren't reused, __jit__-compiled code will perform these operations _in-place_.

```{code-cell} ipython3
:id: tsw2svao8FUp
:outputId: 874acd15-a493-4d63-efe4-9f440d5d2a12

print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = index_add(jax_array, index[::2, 3:], 7.)
print("new array post-addition:")
print(new_jax_array)
```

+++ {"id": "oZ_jE2WAypdL"}

## 🔪 Out-of-Bounds Indexing

+++ {"id": "btRFwEVzypdN"}

In Numpy, you are used to errors being thrown when you index an array outside of its bounds, like this:

```{code-cell} ipython3
:id: 5_ZM-BJUypdO
:outputId: 461f38cd-9452-4bcc-a44f-a07ddfa12f42
:tags: [raises-exception]

try:
  np.arange(10)[11]
except Exception as e:
  print("Exception {}".format(e))
```

+++ {"id": "eoXrGARWypdR"}

However, raising an error from code running on an accelerator can be difficult or impossible. Therefore, JAX must choose some non-error behavior for out of bounds indexing (akin to how invalid floating point arithmetic results in `NaN`). When the indexing operation is an array index update (e.g. `index_add` or `scatter`-like primitives), updates at out-of-bounds indices will be skipped; when the operation is an array index retrieval (e.g. NumPy indexing or `gather`-like primitives) the index is clamped to the bounds of the array since __something__ must be returned. For example, the last value of the array will be returned from this indexing operation:

```{code-cell} ipython3
:id: cusaAD0NypdR
:outputId: 48428ad6-6cde-43ad-c12d-2eb9b9fe59cf

jnp.arange(10)[11]
```

Note that due to this behavior for index retrieval, functions like `jnp.nanargmin` and `jnp.nanargmax` return -1 for slices consisting of NaNs whereas Numpy would throw an error.

Note also that, as the two behaviors described above are not inverses of each other, reverse-mode automatic differentiation (which turns index updates into index retrievals and vice versa) [will not preserve the semantics of out of bounds indexing](https://github.com/google/jax/issues/5760). Thus it may be a good idea to think of out-of-bounds indexing in JAX as a case of [undefined behavior](https://en.wikipedia.org/wiki/Undefined_behavior).

+++ {"id": "LwB07Kx5sgHu"}

## 🔪 Non-array inputs: NumPy vs. JAX

NumPy is generally happy accepting Python lists or tuples as inputs to its API functions:

```{code-cell} ipython3
---
id: sErQES14sjCG
outputId: 6bc29168-624a-4d51-eef1-220aeaf49985
---
np.sum([1, 2, 3])
```

+++ {"id": "ZJ1Wt1bTtrSA"}

JAX departs from this, generally returning a helpful error:

```{code-cell} ipython3
---
id: DFEGcENSsmEc
outputId: 86105261-0aec-41e0-c8a6-16eec437e2a8
---
try:
  jnp.sum([1, 2, 3])
except TypeError as e:
  print(f"TypeError: {e}")
```

+++ {"id": "QPliLUZztxJt"}

This is a deliberate design choice, because passing lists or tuples to traced functions can lead to silent performance degradation that might otherwise be difficult to detect.

For example, consider the following permissive version of `jnp.sum` that allows list inputs:

```{code-cell} ipython3
---
id: jhe-L_TwsvKd
outputId: 24ef84d4-79e5-42de-f8d4-34e6701c2576
---
def permissive_sum(x):
  return jnp.sum(jnp.array(x))

x = list(range(10))
permissive_sum(x)
```

+++ {"id": "m0XZLP7nuYdE"}

The output is what we would expect, but this hides potential performance issues under the hood. In JAX's tracing and JIT compilation model, each element in a Python list or tuple is treated as a separate JAX variable, and individually processed and pushed to device. This can be seen in the jaxpr for the ``permissive_sum`` function above:

```{code-cell} ipython3
---
id: k81u6DQ7vAjQ
outputId: 52847378-ba8c-4e84-fb8b-dabbaded6a00
---
make_jaxpr(permissive_sum)(x)
```

+++ {"id": "C0_dpCfpvCts"}

Each entry of the list is handled as a separate input, resulting in a tracing & compilation overhead that grows linearly with the size of the list. To prevent surprises like this, JAX avoids implicit conversions of lists and tuples to arrays.

If you would like to pass a tuple or list to a JAX function, you can do so by first explicitly converting it to an array:

```{code-cell} ipython3
---
id: nFf_DydixG8v
outputId: 5e4392b6-37eb-4a24-ce4f-43518e61d9b1
---
jnp.sum(jnp.array(x))
```

+++ {"id": "MUycRNh6e50W"}

## 🔪 Random Numbers

+++ {"id": "O8vvaVt3MRG2"}

> _If all scientific papers whose results are in doubt because of bad 
> `rand()`s were to disappear from library shelves, there would be a 
> gap on each shelf about as big as your fist._ - Numerical Recipes

+++ {"id": "Qikt9pPW9L5K"}

### RNGs and State
You're used to _stateful_ pseudorandom number generators (PRNGs) from numpy and other libraries, which helpfully hide a lot of details under the hood to give you a ready fountain of pseudorandomness:

```{code-cell} ipython3
:id: rr9FeP41fynt
:outputId: 849d84cf-04ad-4e8b-9505-a92f6c0d7a39

print(np.random.random())
print(np.random.random())
print(np.random.random())
```

+++ {"id": "ORMVVGZJgSVi"}

Underneath the hood, numpy uses the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) PRNG to power its pseudorandom functions.  The PRNG has a period of $2^{19937}-1$ and at any point can be described by __624 32bit unsigned ints__ and a __position__ indicating how much of this  "entropy" has been used up.

```{code-cell} ipython3
:id: 7Pyp2ajzfPO2

np.random.seed(0)
rng_state = np.random.get_state()
#print(rng_state)
# --> ('MT19937', array([0, 1, 1812433255, 1900727105, 1208447044,
#       2481403966, 4042607538,  337614300, ... 614 more numbers..., 
#       3048484911, 1796872496], dtype=uint32), 624, 0, 0.0)
```

+++ {"id": "aJIxHVXCiM6m"}

This pseudorandom state vector is automagically updated behind the scenes every time a random number is needed, "consuming" 2 of the uint32s in the Mersenne twister state vector:

```{code-cell} ipython3
:id: GAHaDCYafpAF

_ = np.random.uniform()
rng_state = np.random.get_state()
#print(rng_state) 
# --> ('MT19937', array([2443250962, 1093594115, 1878467924,
#       ..., 2648828502, 1678096082], dtype=uint32), 2, 0, 0.0)

# Let's exhaust the entropy in this PRNG statevector
for i in range(311):
  _ = np.random.uniform()
rng_state = np.random.get_state()
#print(rng_state) 
# --> ('MT19937', array([2443250962, 1093594115, 1878467924,
#       ..., 2648828502, 1678096082], dtype=uint32), 624, 0, 0.0)

# Next call iterates the RNG state for a new batch of fake "entropy".
_ = np.random.uniform()
rng_state = np.random.get_state()
# print(rng_state) 
# --> ('MT19937', array([1499117434, 2949980591, 2242547484, 
#      4162027047, 3277342478], dtype=uint32), 2, 0, 0.0)
```

+++ {"id": "N_mWnleNogps"}

The problem with magic PRNG state is that it's hard to reason about how it's being used and updated across different threads, processes, and devices, and it's _very easy_ to screw up when the details of entropy production and consumption are hidden from the end user.

The Mersenne Twister PRNG is also known to have a [number](https://cs.stackexchange.com/a/53475) of problems, it has a large 2.5Kb state size, which leads to problematic [initialization issues](https://dl.acm.org/citation.cfm?id=1276928).  It [fails](http://www.pcg-random.org/pdf/toms-oneill-pcg-family-v1.02.pdf) modern BigCrush tests, and is generally slow.

+++ {"id": "Uvq7nV-j4vKK"}

### JAX PRNG

+++ {"id": "COjzGBpO4tzL"}

JAX instead implements an _explicit_ PRNG where entropy production and consumption are handled by explicitly passing and iterating PRNG state.  JAX uses a modern [Threefry counter-based PRNG](https://github.com/google/jax/blob/master/design_notes/prng.md) that's __splittable__.  That is, its design allows us to __fork__ the PRNG state into new PRNGs for use with parallel stochastic generation.

The random state is described by two unsigned-int32s that we call a __key__:

```{code-cell} ipython3
:id: yPHE7KTWgAWs
:outputId: 329e7757-2461-434c-a08c-fde80a2d10c9

from jax import random
key = random.PRNGKey(0)
key
```

+++ {"id": "XjYyWYNfq0hW"}

JAX's random functions produce pseudorandom numbers from the PRNG state, but __do not__ change the state!  

Reusing the same state will cause __sadness__ and __monotony__, depriving the enduser of __lifegiving chaos__:

```{code-cell} ipython3
:id: 7zUdQMynoE5e
:outputId: 50617324-b887-42f2-a7ff-2a10f92d876a

print(random.normal(key, shape=(1,)))
print(key)
# No no no!
print(random.normal(key, shape=(1,)))
print(key)
```

+++ {"id": "hQN9van8rJgd"}

Instead, we __split__ the PRNG to get usable __subkeys__ every time we need a new pseudorandom number:

```{code-cell} ipython3
:id: ASj0_rSzqgGh
:outputId: bcc2ed60-2e41-4ef8-e84f-c724654aa198

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    \---SPLIT --> new key   ", key)
print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
```

+++ {"id": "tqtFVE4MthO3"}

We propagate the __key__ and make new __subkeys__ whenever we need a new random number:

```{code-cell} ipython3
:id: jbC34XLor2Ek
:outputId: 6834a812-7160-4646-ee19-a246f683905a

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    \---SPLIT --> new key   ", key)
print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
```

+++ {"id": "0KLYUluz3lN3"}

We can generate more than one __subkey__ at a time:

```{code-cell} ipython3
:id: lEi08PJ4tfkX
:outputId: 3bb513de-8d14-4d37-ae57-51d6f5eaa762

key, *subkeys = random.split(key, 4)
for subkey in subkeys:
  print(random.normal(subkey, shape=(1,)))
```

+++ {"id": "rg4CpMZ8c3ri"}

## 🔪 Control Flow

+++ {"id": "izLTvT24dAq0"}

### ✔ python control_flow + autodiff ✔

If you just want to apply `grad` to your python functions, you can use regular python control-flow constructs with no problems, as if you were using [Autograd](https://github.com/hips/autograd) (or Pytorch or TF Eager).

```{code-cell} ipython3
:id: aAx0T3F8lLtu
:outputId: 808cfa77-d924-4586-af19-35a8fd7d2238

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

print(grad(f)(2.))  # ok!
print(grad(f)(4.))  # ok!
```

+++ {"id": "hIfPT7WMmZ2H"}

### python control flow + JIT

Using control flow with `jit` is more complicated, and by default it has more constraints.

This works:

```{code-cell} ipython3
:id: OZ_BJX0CplNC
:outputId: 48ce004c-536a-44f5-b020-9267825e7e4d

@jit
def f(x):
  for i in range(3):
    x = 2 * x
  return x

print(f(3))
```

+++ {"id": "22RzeJ4QqAuX"}

So does this:

```{code-cell} ipython3
:id: pinVnmRWp6w6
:outputId: e3e6f2f7-ba59-4a98-cdfc-905c91b38ed1

@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]):
    y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

+++ {"id": "TStltU2dqf8A"}

But this doesn't, at least by default:

```{code-cell} ipython3
:id: 9z38AIKclRNM
:outputId: 466730dd-df8b-4b80-ac5e-e55b5ea85ec7

@jit
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

# This will fail!
try:
  f(2)
except Exception as e:
  print("Exception {}".format(e))
```

+++ {"id": "pIbr4TVPqtDN"}

__What gives!?__

When we `jit`-compile a function, we usually want to compile a version of the function that works for many different argument values, so that we can cache and reuse the compiled code. That way we don't have to re-compile on each function evaluation.

For example, if we evaluate an `@jit` function on the array `jnp.array([1., 2., 3.], jnp.float32)`, we might want to compile code that we can reuse to evaluate the function on `jnp.array([4., 5., 6.], jnp.float32)` to save on compile time.

To get a view of your Python code that is valid for many different argument values, JAX traces it on _abstract values_ that represent sets of possible inputs. There are [multiple different levels of abstraction](https://github.com/google/jax/blob/master/jax/_src/abstract_arrays.py), and different transformations use different abstraction levels.

By default, `jit` traces your code on the `ShapedArray` abstraction level, where each abstract value represents the set of all array values with a fixed shape and dtype. For example, if we trace using the abstract value `ShapedArray((3,), jnp.float32)`, we get a view of the function that can be reused for any concrete value in the corresponding set of arrays. That means we can save on compile time.

But there's a tradeoff here: if we trace a Python function on a `ShapedArray((), jnp.float32)` that isn't committed to a specific concrete value, when we hit a line like `if x < 3`, the expression `x < 3` evaluates to an abstract `ShapedArray((), jnp.bool_)` that represents the set `{True, False}`. When Python attempts to coerce that to a concrete `True` or `False`, we get an error: we don't know which branch to take, and can't continue tracing! The tradeoff is that with higher levels of abstraction we gain a more general view of the Python code (and thus save on re-compilations), but we require more constraints on the Python code to complete the trace.

The good news is that you can control this tradeoff yourself. By having `jit` trace on more refined abstract values, you can relax the traceability constraints. For example, using the `static_argnums` argument to `jit`, we can specify to trace on concrete values of some arguments. Here's that example function again:

```{code-cell} ipython3
:id: -Tzp0H7Bt1Sn
:outputId: aba57a88-d8eb-40b0-ff22-7c266d892b13

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jit(f, static_argnums=(0,))

print(f(2.))
```

+++ {"id": "MHm1hIQAvBVs"}

Here's another example, this time involving a loop:

```{code-cell} ipython3
:id: iwY86_JKvD6b
:outputId: 1ec847ea-df2b-438d-c0a1-fabf7b93b73d

def f(x, n):
  y = 0.
  for i in range(n):
    y = y + x[i]
  return y

f = jit(f, static_argnums=(1,))

f(jnp.array([2., 3., 4.]), 2)
```

+++ {"id": "nSPTOX8DvOeO"}

In effect, the loop gets statically unrolled.  JAX can also trace at _higher_ levels of abstraction, like `Unshaped`, but that's not currently the default for any transformation

+++ {"id": "wWdg8LTYwCW3"}

️⚠️ **functions with argument-__value__ dependent shapes**

These control-flow issues also come up in a more subtle way: numerical functions we want to __jit__ can't specialize the shapes of internal arrays on argument _values_ (specializing on argument __shapes__ is ok).  As a trivial example, let's make a function whose output happens to depend on the input variable `length`.

```{code-cell} ipython3
:id: Tqe9uLmUI_Gv
:outputId: fe319758-9959-434c-ab9d-0926e599dbc0

def example_fun(length, val):
  return jnp.ones((length,)) * val
# un-jit'd works fine
print(example_fun(5, 4))

bad_example_jit = jit(example_fun)
# this will fail:
try:
  print(bad_example_jit(10, 4))
except Exception as e:
  print("Exception {}".format(e))
# static_argnums tells JAX to recompile on changes at these argument positions:
good_example_jit = jit(example_fun, static_argnums=(0,))
# first compile
print(good_example_jit(10, 4))
# recompiles
print(good_example_jit(5, 4))
```

+++ {"id": "MStx_r2oKxpp"}

`static_argnums` can be handy if `length` in our example rarely changes, but it would be disastrous if it changed a lot!  

Lastly, if your function has global side-effects, JAX's tracer can cause weird things to happen. A common gotcha is trying to print arrays inside __jit__'d functions:

```{code-cell} ipython3
:id: m2ABpRd8K094
:outputId: 64da37a0-aa06-46a3-e975-88c676c5b9fa

@jit
def f(x):
  print(x)
  y = 2 * x
  print(y)
  return y
f(2)
```

+++ {"id": "uCDcWG4MnVn-"}

### Structured control flow primitives

There are more options for control flow in JAX. Say you want to avoid re-compilations but still want to use control flow that's traceable, and that avoids un-rolling large loops. Then you can use these 4 structured control flow primitives:

 - `lax.cond`  _differentiable_
 - `lax.while_loop` __fwd-mode-differentiable__
 - `lax.fori_loop` __fwd-mode-differentiable__
 - `lax.scan` _differentiable_

+++ {"id": "Sd9xrLMXeK3A"}

#### cond
python equivalent:

```python
def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)
```

```{code-cell} ipython3
:id: SGxz9JOWeiyH
:outputId: b29da06c-037f-4b05-dbd8-ba52ac35a8cf

from jax import lax

operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
# --> array([1.], dtype=float32)
lax.cond(False, lambda x: x+1, lambda x: x-1, operand)
# --> array([-1.], dtype=float32)
```

+++ {"id": "xkOFAw24eOMg"}

#### while_loop

python equivalent:
```
def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
```

```{code-cell} ipython3
:id: jM-D39a-c436
:outputId: b9c97167-fecf-4559-9ca7-1cb0235d8ad2

init_val = 0
cond_fun = lambda x: x<10
body_fun = lambda x: x+1
lax.while_loop(cond_fun, body_fun, init_val)
# --> array(10, dtype=int32)
```

+++ {"id": "apo3n3HAeQY_"}

#### fori_loop
python equivalent:
```
def fori_loop(start, stop, body_fun, init_val):
  val = init_val
  for i in range(start, stop):
    val = body_fun(i, val)
  return val
```

```{code-cell} ipython3
:id: dt3tUpOmeR8u
:outputId: 864f2959-2429-4666-b364-4baf90a57482

init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x+i
lax.fori_loop(start, stop, body_fun, init_val)
# --> array(45, dtype=int32)
```

+++ {"id": "SipXS5qiqk8e"}

#### Summary

$$
\begin{array} {r|rr} 
\hline \
\textrm{construct} 
& \textrm{jit} 
& \textrm{grad} \\
\hline \
\textrm{if} & ❌ & ✔ \\
\textrm{for} & ✔* & ✔\\
\textrm{while} & ✔* & ✔\\
\textrm{lax.cond} & ✔ & ✔\\
\textrm{lax.while_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.fori_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.scan} & ✔ & ✔\\
\hline
\end{array}
$$
<center>$\ast$ = argument-__value__-independent loop condition - unrolls the loop </center>

+++ {"id": "DKTMw6tRZyK2"}

## 🔪 NaNs

+++ {"id": "ncS0NI4jZrwy"}

### Debugging NaNs

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by:

* setting the `JAX_DEBUG_NANS=True` environment variable;

* adding `from jax.config import config` and `config.update("jax_debug_nans", True)` near the top of your main file;

* adding `from jax.config import config` and `config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

This will cause computations to error-out immediately on production of a NaN. Switching this option on adds a nan check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@jit`. For code under an `@jit`, the output of every `@jit` function is checked and if a nan is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jit` at a time.

There could be tricky situations that arise, like nans that only occur under a `@jit` but don't get produced in de-optimized mode. In that case you'll see a warning message print out but your code will continue to execute.

If the nans are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple jaxpr interpreter that walks the sequence of primitive operations in reverse. In the example below, we started an ipython repl with the command line `env JAX_DEBUG_NANS=True ipython`, then ran this:

+++

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
    107           return DeviceArray(device_buffer, *result_shape)

FloatingPointError: invalid value
```

+++

The nan generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@jit`, as the example below shows.

+++

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

+++

When this code sees a nan in the output of an `@jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

⚠️ You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lots of device-host round-trips and performance regressions!

+++ {"id": "YTktlwTTMgFl"}

## Double (64bit) precision

At the moment, JAX by default enforces single-precision numbers to mitigate the Numpy API's tendency to aggressively promote operands to `double`.  This is the desired behavior for many machine-learning applications, but it may catch you by surprise!

```{code-cell} ipython3
:id: CNNGtzM3NDkO
:outputId: d1384021-d9bf-450f-a9ae-82024fa5fc1a

x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
x.dtype
```

+++ {"id": "VcvqzobxNPbd"}

To use double-precision numbers, you need to set the `jax_enable_x64` configuration variable __at startup__.  

There are a few ways to do this:

1. You can enable 64bit mode by setting the environment variable `JAX_ENABLE_X64=True`.

2. You can manually set the `jax_enable_x64` configuration flag at startup:

   ```python
   # again, this only works on startup!
   from jax.config import config
   config.update("jax_enable_x64", True)
   ```

3. You can parse command-line flags with `absl.app.run(main)`

   ```python
   from jax.config import config
   config.config_with_absl()
   ```

4. If you want JAX to run absl parsing for you, i.e. you don't want to do `absl.app.run(main)`, you can instead use

   ```python
   from jax.config import config
   if __name__ == '__main__':
     # calls config.config_with_absl() *and* runs absl parsing
     config.parse_flags_with_absl()
   ```

Note that #2-#4 work for _any_ of JAX's configuration options.

We can then confirm that `x64` mode is enabled:

```{code-cell} ipython3
:id: HqGbBa9Rr-2g
:outputId: cd241d63-3d00-4fd7-f9c0-afc6af01ecf4

import jax.numpy as jnp
from jax import random
x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
x.dtype # --> dtype('float64')
```

+++ {"id": "6Cks2_gKsXaW"}

### Caveats
⚠️ XLA doesn't support 64-bit convolutions on all backends!

+++ {"id": "WAHjmL0E2XwO"}

## Fin.

If something's not covered here that has caused you weeping and gnashing of teeth, please let us know and we'll extend these introductory _advisos_!
