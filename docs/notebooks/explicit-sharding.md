---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "ZVJCNxUcVkkm"}

# Explicit sharding (a.k.a. "sharding in types")

+++ {"id": "ATLBMlw3VcCJ"}

JAX's traditional automatic sharding leaves sharding decisions to the compiler.
You can provide hints to the compiler using
`jax.lax.with_sharding_constraint` but for the most part you're supposed to be
focussed on the math while the compiler worries about sharding.

But what if you have a strong opinion about how you want your program sharded?
With enough calls to `with_sharding_constraint` you can probably guide the
compiler's hand to make it do what you want. But "compiler tickling" is
famously not a fun programming model. Where should you put the sharding
constraints? You could put them on every single intermediate but that's a lot
of work and it's also easy to make mistakes that way because there's no way to
check that the shardings make sense together. More commonly, people add just
enough sharding annotations to constrain the compiler. But this is a slow
iterative process. It's hard to know ahead of time what XLA's GSPMD pass will
do (it's a whole-program optimization) so all you can do is add annotations,
inspect XLA's sharding choices to see what happened, and repeat.

To fix this we've come up with a different style of sharding programming we
call "explicit sharding" or "sharding in types". The idea is that sharding
propagation happens at the JAX level at trace time. Each JAX operation has a
sharding rule that takes the shardings of the op's arguments and produces a
sharding for the op's result. For most operations these rules are simple and
obvious because there's only one reasonable choice. But for some operations it's
unclear how to shard the result. In that case we ask the programmer
to provide an `out_sharding` argument explicitly and we throw a (trace-time)
error otherwise. Since the shardings are propagated at trace time they can
also be _queried_ at trace time too. In the rest of this doc we'll describe
how to use explicit sharding mode. Note that this is a new feature so we
expect there to be bugs and unimplemented cases. Please let us know when you
find something that doesn't work!

```{code-cell} ipython3
:id: hVi6mApuVw3r

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType, set_mesh, get_abstract_mesh
from jax.experimental.shard import reshard, auto_axes, explicit_axes

jax.config.update('jax_num_cpu_devices', 8)
```

+++ {"id": "oU5O6yOLWqbP"}

## Setting up an explicit mesh

The main idea behind explicit shardings, (a.k.a. sharding-in-types), is that
the JAX-level _type_ of a value includes a description of how the value is sharded.
We can query the JAX-level type of any JAX value (or Numpy array, or Python
scalar) using `jax.typeof`:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: mzDIDvj7Vw0k
outputId: 09ef049b-461f-47db-bf58-dc10b42fe40a
---
some_array = np.arange(8)
print(f"JAX-level type of some_array: {jax.typeof(some_array)}")
```

+++ {"id": "TZzp_1sXW061"}

Importantly, we can query the type even while tracing under a `jit` (the JAX-level type
is almost _defined_ as "the information about a value we have access to while
under a jit).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: IyPx_-IBVwxr
outputId: 0cd3122f-e579-45d7-868d-e42bb0eacddb
---
@jax.jit
def foo(x):
  print(f"JAX-level type of x during tracing: {jax.typeof(x)}")
  return x + x

foo(some_array)
```

+++ {"id": "c3gNPzfZW45K"}

These types show the shape and dtype of array but they don't appear to
show sharding. (Actually, they _did_ show sharding, but the shardings were
trivial. See "Concrete array shardings", below.) To start seeing some
interesting shardings we need to set up an explicit-sharding mesh. We use
`set_mesh` to set it as the current mesh for the remainder of this notebook.
(If you only want to set the mesh for some particular scope and return to the previous
mesh afterwards then you can use the context manager `jax.sharding.use_mesh` instead.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: NO2ulM_QW7a8
outputId: d888371b-080e-4bff-be5d-ea56beda3aac
---
mesh = jax.make_mesh((2, 4), ("X", "Y"),
                     axis_types=(AxisType.Explicit, AxisType.Explicit))
set_mesh(mesh)

print(f"Current mesh is: {get_abstract_mesh()}")
```

+++ {"id": "V7bVz6tzW_Eb"}

Now we can create some sharded arrays using `reshard`:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1-TzmA0AXCAf
outputId: 1c7cc3ac-4b0e-42b7-facc-c706af10d7d2
---
replicated_array = np.arange(8).reshape(4, 2)
sharded_array = reshard(replicated_array, P("X", None))

print(f"replicated_array type: {jax.typeof(replicated_array)}")
print(f"sharded_array type: {jax.typeof(sharded_array)}")
```

+++ {"id": "B0jBBXtgXBxr"}

We should read the type `f32[4@X, 2]` as "a 4-by-2 array of 32-bit floats whose first dimension
is sharded along mesh axis 'X'. The array is replicated along all other mesh
axes"

+++ {"id": "N8yMauHAXKtX"}

These shardings associated with JAX-level types propagate through operations. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Gy7ABds3XND3
outputId: 0d72dad2-381a-4e96-f771-40d705da1376
---
arg0 = reshard(np.arange(4).reshape(4, 1), P("X", None))
arg1 = reshard(np.arange(8).reshape(1, 8), P(None, "Y"))

result = arg0 + arg1

print(f"arg0 sharding: {jax.typeof(arg0)}")
print(f"arg1 sharding: {jax.typeof(arg1)}")
print(f"result sharding: {jax.typeof(result)}")
```

+++ {"id": "lwsygUmVXPCk"}

We can do the same type querying under a jit:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: grCcotr-XQjY
outputId: c2db656c-809f-49a6-c948-629d6420360c
---
@jax.jit
def add_arrays(x, y):
  ans = x + y
  print(f"x sharding: {jax.typeof(x)}")
  print(f"y sharding: {jax.typeof(y)}")
  print(f"ans sharding: {jax.typeof(ans)}")
  return ans

add_arrays(arg0, arg1)
```

+++ {"id": "lVd6a5ufXZoH"}

That's the gist of it. Shardings propagate deterministically at trace time and
we can query them at trace time.

+++ {"id": "ETtwK3LCXSkd"}

## Sharding rules and operations with ambiguous sharding

Each op has a sharding rule which specifies its output sharding given its
input shardings. A sharding rule may also throw a (trace-time) error. Each op
is free to implement whatever sharding rule it likes, but the usual pattern is
the following: For each output axis we identify zero of more corresponding
input axes. The output axis is then
sharded according to the “consensus” sharding of the corresponding input axes. i.e., it's
`None` if the input shardings are all `None`, and it's the common non-None input sharding
if there’s exactly one of them, or an error (requiring an explicit out_sharding=... kwarg) otherwise.

+++ {"id": "an8-Fq1uXehp"}

This procedure is done on an axis-by-axis basis. When it’s done, we might end
up with an array sharding that mentions a mesh axis more than once, which is
illegal. In that case we raise a (trace-time) sharding error and ask for an
explicit out_sharding.

Here are some example sharding rules:
   * nullary ops like `jnp.zeros`, `jnp.arange`: These ops create arrays out of whole
     cloth so they don’t have input shardings to propagate. Their output is
     unsharded by default unless overridden by the out_sharding kwarg.
   * unary elementwise ops like `sin`, `exp`: The output is sharded the same as the
     input.
   * binary ops (`+`, `-`, `*` etc.): Axis shardings of “zipped” dimensions
     must match (or be `None`). “Outer product” dimensions (dimensions that
     appear in only one argument) are sharded as they are in the input. If the
     result ends up mentioning a mesh axis more than once it's an error.
   * `reshape.` Reshape is a particularly tricky op. An output axis can map to more
     than one input axis (when reshape is used to merge axes) or just a part
     of an input axis (when reshape is used to split axes). Our usual rules
     don’t apply. Instead we treat reshape as follows. We strip away singleton
     axes (these can’t be sharded anyway. Then
     we decide whether the reshape is a “split” (splitting a single axis into
     two or more adjacent axes), a “merge” (merging two or more adjacent axes
     into a single one) or something else. If we have a split or merge case in
     which the split/merged axes are sharded as None then we shard the
     resulting split/merged axes as None and the other axes according to their
     corresponding input axis shardings. In all other cases we throw an error
     and require the user to provide an `out_sharding` argument.

+++ {"id": "jZMp6w48Xmd7"}

## JAX transformations and higher-order functions

The staged-out representation of JAX programs is explicitly typed. (We call
the types “avals” but that’s not important.) In explicit-sharding mode, the
sharding is part of that type. This means that shardings need to match
wherever types need to match. For example, the two sides of a `lax.cond` need to
have results with matching shardings. And the carry of `lax.scan` needs to have the
same sharding at the input and the output of the scan body. And when you
construct a jaxpr without concrete arguments using `make_jaxpr` you need to
provide shardings too. Certain JAX transformations perform type-level
operations. Automatic differentation constructs a tangent type for each primal
type in the original computation (e.g. `TangentOf(float) == float`,
`TangentOf(int) == float0`). With sharding in the types, this means that tangent
values are sharded in the same way as their primal values. Vmap and scan also
do type-level operations, they lift an array shape to a rank-augmented version
of that shape. That extra array axis needs a sharding. We can infer it from the
arguments to the vmap/scan but they all need to agree. And a nullary vmap/scan
needs an explicit sharding argument just as it needs an explicit length
argument.

+++ {"id": "ERJx4p0tXoS3"}

## Working around unimplemented sharding rules using `auto_sharding`

The implementation of explicit sharding is still a work-in-progress and there
are plenty of ops that are missing sharding rules. For example, `scatter` and
`gather` (i.e. indexing ops).

Normally we wouldn't suggest using a feature with so many unimplemented cases,
but in this instance there's a reasonable fallback you can use: `auto_axes`.
The idea is that you can temporarily drop into a context where the mesh axes
are "auto" rather than "explicit". You explicitly specify how you intend the
final result of the `auto_axes` to be sharded as it gets returned to the calling context.

This works as a fallback for ops with unimplemented sharding rules. It also
works when you want to override the sharding-in-types type system. For
example, suppose we want to add a `f32[4@X, 4]` to a `f32[4, 4@X]`. Our
sharding rule for addition would throw an error: the result would need to be
`f32[4@X, 4@X]`, which tries uses a mesh axis twice, which is illegal. But say you
want to perform the operation anyway, and you want the result to be sharded along
the first axis only, like `f32[4@X, 4]`. You can do this as follows:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: fpFEaMBcXsJG
outputId: 5b84b1d1-d7b2-4e9a-ba98-3dd34a5465ef
---
some_x = reshard(np.arange(16).reshape(4, 4), P("X", None))
some_y = reshard(np.arange(16).reshape(4, 4), P(None, "X"))

try:
  some_x + some_y
except Exception as e:
  print("ERROR!")
  print(e)

print("=== try again with auto_axes ===")

@auto_axes
def add_with_out_sharding_kwarg(x, y):
  print(f"We're in auto-sharding mode here. This is the current mesh: {get_abstract_mesh()}")
  return x + y

result = add_with_out_sharding_kwarg(some_x, some_y, out_sharding=P("X", None))
print(f"Result type: {jax.typeof(result)}")
```

+++ {"id": "8-_zDr-AXvb6"}

## Using a mixture of sharding modes

JAX now has three styles of parallelism:

 * *Automatic sharding* is where you treat all the devices as a single logical
   machine and write a "global view" array program for that machine. The
   compiler decides how to partition the data and computation across the
   available devices. You can give hints to the compiler using
   `with_sharding_constraint`.
 * *Explicit Sharding* (\*new\*) is similar to automatic sharding in that
   you're writing a global-view program. The difference is that the sharding
   of each array is part of the array's JAX-level type making it an explicit
   part of the programming model. These shardings are propagated at the JAX
   level and queryable at trace time. It's still the compiler's responsibility
   to turn the whole-array program into per-device programs (turning `jnp.sum`
   into `psum` for example) but the compiler is heavily constrained by the
   user-supplied shardings.
 * *Manual Sharding* (`shard_map`) is where you write a program from the
   perspective of a single device. Communication between devices happens via
   explicit collective operations like psum.

A summary table:

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|---|---|---|---|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

The current mesh tells us which sharding mode we're in. We can query it with
`get_abstract_mesh`:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: geptWrdYX0OM
outputId: b8c3813f-60bb-4ccf-9da7-73462c57963f
---
print(f"Current mesh is: {get_abstract_mesh()}")
```

+++ {"id": "AQQjzUeGX4P6"}

Since `axis_types=(Explicit, Explicit)`, this means we're in fully-explicit
mode. Notice that the sharding mode is associated with a mesh _axis_, not the
mesh as a whole. We can actually mix sharding modes by having a different
sharding mode for each mesh axis. Shardings (on JAX-level types) can only
mention _explicit_ mesh axes and collective operations like `psum` can only
mention _manual_ mesh axes.

+++ {"id": "LZWjgiMZ7uSS"}

You can use the `auto_axes` API to be `Auto` over some mesh axes while being `Explicit` over other. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: IVzPSkp77uCF
outputId: db80a604-98ac-4343-8677-23729adf7ffc
---
import functools

@functools.partial(auto_axes, axes='X')
def g(y):
  print(f'mesh inside g: {get_abstract_mesh()}')
  print(f'y.sharding inside g: {jax.typeof(y) = }', end='\n\n')
  return y * 2

@jax.jit
def f(arr1):
  print(f'mesh inside f: {get_abstract_mesh()}')
  x = jnp.sin(arr1)
  print(f'x.sharding: {jax.typeof(x)}', end='\n\n')

  z = g(x, out_sharding=P("X", "Y"))

  print(f'z.sharding: {jax.typeof(z)}', end="\n\n")
  return z + 1

some_x = reshard(np.arange(16).reshape(4, 4), P("X", "Y"))
f(some_x)
```

+++ {"id": "_3sfJjRq8w9f"}

As you can see, inside `g`, the type of `arr1` is `ShapedArray(float32[4,4@Y])` which indicates it's Explicit over `Y` mesh axis while auto over `X`.


You can also use the `explicit_axes` API to drop into `Explicit` mode over some or all mesh axes.

```{code-cell} ipython3
auto_mesh = jax.make_mesh((2, 4), ("X", "Y"),
                           axis_types=(AxisType.Auto, AxisType.Auto))

@functools.partial(explicit_axes, axes=('X', 'Y'))
def explicit_g(y):
  print(f'mesh inside g: {get_abstract_mesh()}')
  print(f'y.sharding inside g: {jax.typeof(y) = }')
  z = y * 2
  print(f'z.sharding inside g: {jax.typeof(z) = }', end='\n\n')
  return z

@jax.jit
def f(arr1):
  print(f'mesh inside f: {get_abstract_mesh()}', end='\n\n')
  x = jnp.sin(arr1)

  z = explicit_g(x, in_sharding=P("X", "Y"))

  return z + 1

with jax.sharding.use_mesh(auto_mesh):
  some_x = jax.device_put(np.arange(16).reshape(4, 4), P("X", "Y"))
  f(some_x)
```

As you can see, all axes of mesh inside `f` are of type `Auto` while inside `g`, they are of type `Explicit`.
Because of that, sharding is visible on the type of arrays inside `g`.

+++ {"id": "sJcWbfAh7UcO"}

## Concrete array shardings can mention `Auto` mesh axis

You can query the sharding of a concrete array `x` with `x.sharding`. You
might expect the result to be the same as the sharding associated with the
value's type, `jax.typeof(x).sharding`. It might not be! The concrete array sharding, `x.sharding`, describes the sharding along
both `Explicit` and `Auto` mesh axes. It's the sharding that the compiler
eventually chose. Whereas the type-specificed sharding,
`jax.typeof(x).sharding`, only describes the sharding along `Explicit` mesh
axes. The `Auto` axes are deliberately hidden from the type because they're
the purview of the compiler. We can think of the concrete array sharding being consistent with, but more specific than,
the type-specified sharding. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ivLl6bxmX7EZ
outputId: 6d7b7fce-68b6-47f1-b214-d62bda8d7b6e
---
def compare_shardings(x):
  print(f"=== with mesh: {get_abstract_mesh()} ===")
  print(f"Concrete value sharding: {x.sharding.spec}")
  print(f"Type-specified sharding: {jax.typeof(x).sharding.spec}")

my_array = jnp.sin(reshard(np.arange(8), P("X")))
compare_shardings(my_array)

@auto_axes
def check_in_auto_context(x):
  compare_shardings(x)
  return x

check_in_auto_context(my_array, out_sharding=P("X"))
```

+++ {"id": "MRFccsi5X8so"}

Notice that at the top level, where we're currently in a fully `Explicit` mesh
context, the concrete array sharding and type-specified sharding agree. But
under the `auto_axes` decorator we're in a fully `Auto` mesh context and the
two shardings disagree: the type-specified sharding is `P(None)` whereas the
concrete array sharding is `P("X")` (though it could be anything! It's up to
the compiler).
