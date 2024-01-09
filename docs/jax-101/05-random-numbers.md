---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "1Op_vnmkjw3z"}

# Pseudo Random Numbers in JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/05-random-numbers.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/jax-101/05-random-numbers.ipynb)

*Authors: Matteo Hessel & Rosalia Schneider*

In this section we focus on pseudo random number generation (PRNG); that is, the process of algorithmically generating sequences of numbers whose properties approximate the properties of sequences of random numbers sampled from an appropriate distribution. 

PRNG-generated sequences are not truly random because they are actually determined by their initial value, which is typically referred to as the `seed`, and each step of random sampling is a deterministic function of some `state` that is carried over from a sample to the next.

Pseudo random number generation is an essential component of any machine learning or scientific computing framework. Generally, JAX strives to be compatible with NumPy, but pseudo random number generation is a notable exception.

To better understand the difference between the approaches taken by JAX and NumPy when it comes to random number generation we will discuss both approaches in this section.

+++ {"id": "6_117sy0CGEU"}

## Random numbers in NumPy

Pseudo random number generation is natively supported in NumPy by the `numpy.random` module.

In NumPy, pseudo random number generation is based on a global `state`.

This can be set to a deterministic initial condition using `random.seed(SEED)`.

```{code-cell} ipython3
:id: qbmCquES5beU

import numpy as np
np.random.seed(0)
```

+++ {"id": "WImNZxJ-7plK"}

You can inspect the content of the state using the following command.

```{code-cell} ipython3
:id: qNO_vG7z7qUb
:outputId: 47817350-83be-40cc-85c3-46419fdbfda0

def print_truncated_random_state():
  """To avoid spamming the outputs, print only part of the state."""
  full_random_state = np.random.get_state()
  print(str(full_random_state)[:460], '...')

print_truncated_random_state()
```

+++ {"id": "nmqx0gJW9CFo"}

The `state` is updated by each call to a random function:

```{code-cell} ipython3
:id: ZqUzvqF1B1TO
:outputId: c1874391-eb8d-43d8-eb8f-c918ed0a0c1a

np.random.seed(0)

print_truncated_random_state()

_ = np.random.uniform()

print_truncated_random_state()
```

+++ {"id": "G1ICXejY_xR0"}

NumPy allows you to sample both individual numbers, or entire vectors of numbers in a single function call. For instance, you may sample a vector of 3 scalars from a uniform distribution by doing:

```{code-cell} ipython3
:id: 6Xqx2e8tAW5d
:outputId: a428facb-cd16-4375-f5c4-8fc601e60169

np.random.seed(0)
print(np.random.uniform(size=3))
```

+++ {"id": "zPfs8tXTAlr7"}

NumPy provides a *sequential equivalent guarantee*, meaning that sampling N numbers in a row individually or sampling a vector of N numbers results in the same pseudo-random sequences:

```{code-cell} ipython3
:id: bZiBZXHW_2wO
:outputId: 3aff9a51-8a19-4737-a7ad-91b23bfc05f8

np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))
```

+++ {"id": "JGCZI9UTl7o4"}

## Random numbers in JAX

JAX's random number generation differs from NumPy's in important ways. The reason is that NumPy's PRNG design makes it hard to simultaneously guarantee a number of desirable properties for JAX, specifically that code must be:

1. reproducible,
2. parallelizable,
3. vectorisable.

We will discuss why in the following. First, we will focus on the implications of a PRNG design based on a global state. Consider the code:

```{code-cell} ipython3
:id: j441y2NCmnbt
:outputId: 77fe84d7-c86e-417a-95b9-d73663ed40fc

import numpy as np

np.random.seed(0)

def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar() + 2 * baz()

print(foo())
```

+++ {"id": "5kVpfSV5n1d7"}

The function `foo` sums two scalars sampled from a uniform distribution.

The output of this code can only satisfy requirement #1 if we assume a specific order of execution for `bar()` and `baz()`, as native Python does.

This doesn't seem to be a major issue in NumPy, as it is already enforced by Python, but it becomes an issue in JAX. 

Making this code reproducible in JAX would require enforcing this specific order of execution. This would violate requirement #2, as JAX should be able to parallelize `bar` and `baz` when jitting as these functions don't actually depend on each other.

To avoid this issue, JAX does not use a global state. Instead, random functions explicitly consume the state, which is referred to as a `key` .

```{code-cell} ipython3
:id: LuaGUVRUvbzQ
:outputId: bbf525d7-d407-49b4-8bee-2cd827846e04

from jax import random

key = random.PRNGKey(42)

print(key)
```

+++ {"id": "XhFpKnW9F2nF"}

A key is just an array of shape `(2,)`.

'Random key' is essentially just another word for 'random seed'. However, instead of setting it once as in NumPy, any call of a random function in JAX requires a key to be specified. Random functions consume the key, but do not modify it. Feeding the same key to a random function will always result in the same sample being generated:

```{code-cell} ipython3
:id: Tc_Tsv06Fz3l
:outputId: 1472ae73-edbf-4163-9992-46781d258014

print(random.normal(key))
print(random.normal(key))
```

+++ {"id": "foUEgtmTesOx"}

**Note:** Feeding the same key to different random functions can result in correlated outputs, which is generally undesirable. 

**The rule of thumb is: never reuse keys (unless you want identical outputs).**

+++ {"id": "T4dOLP0GGJuB"}

In order to generate different and independent samples, you must `split()` the key *yourself* whenever you want to call a random function:

```{code-cell} ipython3
:id: qChuz1C9CSJe
:outputId: f6eb1dc3-d83c-45ef-d90e-5a12d36fa7e6

print("old key", key)
new_key, subkey = random.split(key)
del key  # The old key is discarded -- we must never use it again.
normal_sample = random.normal(subkey)
print(r"    \---SPLIT --> new key   ", new_key)
print(r"             \--> new subkey", subkey, "--> normal", normal_sample)
del subkey  # The subkey is also discarded after use.

# Note: you don't actually need to `del` keys -- that's just for emphasis.
# Not reusing the same values is enough.

key = new_key  # If we wanted to do this again, we would use new_key as the key.
```

+++ {"id": "WKQMJQB6cGhV"}

`split()` is a deterministic function that converts one `key` into several independent (in the pseudorandomness sense) keys. We keep one of the outputs as the `new_key`, and can safely use the unique extra key (called `subkey`) as input into a random function, and then discard it forever.

If you wanted to get another sample from the normal distribution, you would split `key` again, and so on. The crucial point is that you never use the same PRNGKey twice. Since `split()` takes a key as its argument, we must throw away that old key when we split it.

It doesn't matter which part of the output of `split(key)` we call `key`, and which we call `subkey`. They are all pseudorandom numbers with equal status. The reason we use the key/subkey convention is to keep track of how they're consumed down the road. Subkeys are destined for immediate consumption by random functions, while the key is retained to generate more randomness later.

Usually, the above example would be written concisely as

```{code-cell} ipython3
:id: Xkt5OYjHjWiP

key, subkey = random.split(key)
```

+++ {"id": "ULmPVyd9jWSv"}

which discards the old key automatically.

+++ {"id": "dlaAsObh68R1"}

It's worth noting that `split()` can create as many keys as you need, not just 2:

```{code-cell} ipython3
:id: hbHZP2xM7Egf

key, *forty_two_subkeys = random.split(key, num=43)
```

+++ {"id": "Fhu7ejhLB4R_"}

Another difference between NumPy's and JAX's random modules relates to the sequential equivalence guarantee mentioned above.

As in NumPy, JAX's random module also allows sampling of vectors of numbers.
However, JAX does not provide a sequential equivalence guarantee, because doing so would interfere with the vectorization on SIMD hardware (requirement #3 above).

In the example below, sampling 3 values out of a normal distribution individually using three subkeys gives a different result to using giving a single key and specifying `shape=(3,)`:

```{code-cell} ipython3
:id: 4nB_TA54D-HT
:outputId: 2f259f63-3c45-46c8-f597-4e53dc63cb56

key = random.PRNGKey(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.PRNGKey(42)
print("all at once: ", random.normal(key, shape=(3,)))
```

+++ {"id": "_vBAaU2jrWPk"}

Note that contrary to our recommendation above, we use `key` directly as an input to `random.normal()` in the second example. This is because we won't reuse it anywhere else, so we don't violate the single-use principle.
