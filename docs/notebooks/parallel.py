# ---
# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# [![Open in
# Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/parallel.ipynb)

# # Distributed arrays and automatic parallelization
#
# <!--* freshness: { reviewed: '2025-12-02' } *-->
#
# JAX has three styles of multi-device distributed parallelism, which can be
# mixed and composed. They differ in how much the compiler automatically decides
# versus how much is controlled explicitly in the program:
#
#  * **Compiler-based automatic sharding** is where you program as if using a single
#  "global view" machine, and the compiler chooses how to shard data (with some
#  user-provided constraints via `with_sharding_constraint`) and how to
#  partition computation into per-device programs with collectives.
#  * **Explicit sharding and automatic partitioning** is where you still have a
#  global view but data shardings are explicit in JAX types, inspectable using
#  `jax.typeof`. The compiler still partitions the computation.
#  * **Manual per-device programming** is where you have a per-device view of
#  data and computation, and write explicit communication collectives like
#  `jax.lax.psum`.

# | Mode | View? | Explicit sharding? | Explicit Collectives? |
# |---|---|---|---|
# | Auto | Global | ❌ | ❌ |
# | Explicit | Global | ✅ | ❌ |
# | Manual | Per-device | ✅ | ✅ |

# Before getting into details, here's a quick example using explicit mode.
# First, we create a `jax.Array` sharded across multiple devices:

from __future__ import annotations
import enum

import jax
import jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 8)

# +
jax.set_mesh(jax.make_mesh((4, 2), ('X', 'Y')))  # explicit mode by default

x = jnp.arange(8 * 4.).reshape(8, 4)
x = jax.device_put(x, jax.P('X', 'Y'))
print(jax.typeof(x))  # f32[8@X, 4@Y]
# -

jax.debug.visualize_array_sharding(x)

# Next, we'll apply a computation to it and observe that the result values are
# stored across multiple devices too:

y = jnp.sin(x).T
print(jax.typeof(y))  # f32[4@Y, 8@X]


# The `jnp.sin` and transpose computations were automatically parallelized
# across the devices on which the input values (and output values) are stored.
#
# To understand these modes and how to switch among them, we first need to
# understand meshes.
#
# ## A `Mesh` is a grid of devices with named axes
#
# To describe how data and computation are distributed across devices, we first
# organize our devices into a multi-dimensional grid called a `Mesh`.
# Because communication happens along mesh axes, the mesh shape and device order
# can determine communication performance. The mesh should reflect the
# physical connection topology among the devices.
#
# We distinguish between _concrete_ and _abstract_ meshes. An abstract mesh
# comprises only a shape, axis names, and axis types reflecting the **mode** of
# each axis:

# +
class AbstractMesh:
  axis_sizes: tuple[int, ...]
  axis_names: tuple[str, ...]
  axis_types: tuple[AxisType, ...]

class AxisType(enum.Enum):
  Auto = enum.auto()
  Explicit = enum.auto()
  Manual = enum.auto()

# A concrete mesh additionally includes physical device objects with e.g.
# precise coordinates:


# +
import numpy as np

class Mesh:
  devices: np.ndarray[jax.Device]
  axis_names: tuple[str, ...]
  axis_types: tuple[AxisType, ...]

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self.devices.shape


# -

# At the top level of a program (i.e. not under a `jit`) we can create a
# concrete `Mesh` directly [using
# the class constructor](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh),
# which lets us specify the exact device order, or using the `jax.make_mesh`
# helper, which automatically chooses a device order by taking the underlying
# hardware topology into account:

mesh = jax.make_mesh((4, 2), ('X', 'Y'))
print(mesh)

# By default, all mesh axis types are `AxisType.Explicit`.
#
# To avoid threading `mesh` throughout your program, use `jax.set_mesh` to set
# a concrete mesh globally:

jax.set_mesh(mesh)


# You can also use `with jax.set_mesh(mesh): ...` as a context manager. At the
# top level only, the concrete mesh can be queried using `jax.get_mesh() ->
# jax.sharding.Mesh`.
#
# Under a jit, only the abstract mesh can be queried and changed. Use
# `jax.sharding.get_abstract_mesh() -> jax.sharding.AbstractMesh` to query the
# current abstract mesh, and use `with jax.sharding.use_abstract_mesh(m:
# AbstractMesh): ...` to change the abstract mesh within a context. The axis
# sizes, axis names, and axis types can be changed, but the total size of the
# mesh (i.e. the product of the axis sizes) must not change.
#
# We haven't explained shardings yet, but here's a toy example of changing
# abstract meshes inside a `jax.jit`:

# +
@jax.jit
def f(x):
  abstract_mesh = jax.sharding.AbstractMesh((8,), ('A',), (jax.sharding.AxisType.Explicit,))
  with jax.sharding.use_abstract_mesh(abstract_mesh):
    y = jax.reshard(x, jax.P('A', None))
    return y * 2

z = f(x)
print(jax.typeof(z))  # f32[8@A, 4]
# -

# ## A `Sharding` describes how array values are laid out over a `Mesh`
#
# A `jax.sharding.Sharding` describes distributed memory layout. That is, it
# describes how an array's entries are stored in the physical memories of
# different devices, i.e. how it's _sharded_ over devices.
#
# At the top level, every `jax.Array` has an associated `Sharding`, which
# consists of a concrete `Mesh` along with a `jax.sharding.PartitionSpec`
# (aliased to `jax.P`):

print(x.sharding)
jax.debug.visualize_array_sharding(x)

# Here, `PartitionSpec('X', 'Y')` expresses that the first and second axes of
# the array `x` are sharded over the mesh axes 'X' and 'Y', respectively.
# We can see how that translates to physical storage using `addressable_shards`:

for s in x.addressable_shards:
  print(s.device, s.data, sep='\n', end='\n\n')

# We can use `jax.device_put` (or `jax.reshard`) to produce a new array that is
# sharded over the same mesh of devices but with a different layout specified by
# a `jax.P`.
# (`jax.device_put` is a runtime-level API with more features than
# `jax.reshard`.)
# Since we have a mesh in context, via the `jax.set_mesh` above, we can pass
# `jax.P` instances directly to `jax.device_put`:

y = jax.device_put(x, jax.P('Y', 'X'))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

y = jax.device_put(x, jax.P('X', None))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

# Here, because the mesh axis name 'Y' is not mentioned in `jax.P('X', None)`,
# the array is replicated over the mesh axis 'Y'. (As a shorthand, trailing
# `None` placeholders can be omitted, so that P('X', None) here means the same
# thing as P('X'). But it doesn’t hurt to be explicit!)

for s in y.addressable_shards:
  print(s.device, s.data, sep='\n', end='\n\n')

# By using tuples of axis names inside a `PartitionSpec`, we can shard one array
# axis over multiple mesh axes:

y = jax.device_put(x, jax.P(('X', 'Y')))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

# So an array's data can be replicated over a mesh axis, or one of its array
# axes can be sharded over that mesh axis, but there's another possibility too:
# an array can be _unreduced_ over a mesh axis:

y = jax.device_put(x, jax.P('X', None, unreduced={'Y'}))
print(y.sharding)

# Unreduced means that the logical value equals the distributed sum of the
# physical shards' values along that axis:

for s in y.addressable_shards:
  print(s.device, s.data, sep='\n', end='\n\n')

# Unreduced is useful for delaying distributed reductions, especially in
# the context of autodiff. More on that later.
#
# Note that because every array has its own `Sharding` instance, and every
# `Sharding` instance has its own `Mesh` instance, arrays in scope can be
# associated with different meshes. To illustrate, we can use `jax.device_put`
# with a full `jax.NamedSharding` instance argument rather than using the
# in-context mesh:

mesh2 = jax.make_mesh((8,), ('A',))
z = jax.device_put(x, jax.NamedSharding(mesh2, jax.P('A', None)))
print(z.sharding)
print(y.sharding)

# Now that we understand mesh shapes, axis names, and shardings at the top
# level, we can dive into mesh axis types and how Explicit and Auto modes
# differ.
#
# ## Explicit sharding mode makes sharding queryable at trace-time
#
# In explicit sharding mode, shardings are always queryable via `jax.typeof`,
# even under a `jax.jit`:

print(jax.typeof(x).sharding)

jax.jit(lambda x: print(jax.typeof(x).sharding))(x)

# We also call this mode "sharding in types".
#
# In terms of the printed representation, the type language is roughly:

# + active=""
#  <array_type> ::= <dtype>[<size_and_sharding>, ...]
#  <size_and_sharding> ::= <size> | <size>@<MeshAxisName>
# -

# Where
#  * The mesh axis names in scope are those from `jax.typeof(x).sharding.mesh`
#  * Each mesh axis name must be of Explicit axis type
#  * Each mesh axis name can be mentioned at most once in an array type
#
# These shardings associated with JAX-level types propagate through operations.
# For example:

# +
arg0 = jax.device_put(np.arange(4).reshape(4, 1), jax.P("X", None))
arg1 = jax.device_put(np.arange(8).reshape(1, 8), jax.P(None, "Y"))

result = arg0 + arg1

print(f"{jax.typeof(arg0)=!s}")
print(f"{jax.typeof(arg1)=!s}")
print(f"{jax.typeof(result)=!s}")


# -

# We can do the same type querying under a `jit`:

# +
@jax.jit
def add_arrays(x, y):
  ans = x + y
  print(f"{jax.typeof(arg0)=!s}")
  print(f"{jax.typeof(arg1)=!s}")
  print(f"{jax.typeof(result)=!s}")
  return ans

add_arrays(arg0, arg1)
# -

# Given the input and output shardings, the computation itself is automatically
# partitioned over devices. The compiler inserts communication operations as
# needed. For example:

x = jax.random.normal(jax.random.key(0), (8, 4),
                      out_sharding=jax.P('X', 'Y'))
print(jax.typeof(x))

y = x.sum(0)
print(jax.typeof(y))

# Here, when partitioning the computation, the compiler automatically inserts
# communication collectives to perform the reduction:

compile_txt = jax.jit(lambda x: x.sum(0)).lower(x).compile().as_text()
print('all-reduce(' in compile_txt)

# ### Result shardings follow simple rules, or error and require annotation
#
# Each primitive operation has a sharding propagation rule to determine the
# sharding of the result as a function of input shardings. If there is not an
# obvious output sharding, an error is raised. The goal is to get important
# parallelism decisions in your face, rather than hide them so you might
# accidentally miss them. Put another way, sharding propagation rules prefer to
# error and require annotation rather than falling back to arbitrarily chosen
# defaults.
#
# Each op is able to implement its own sharding propagation rule, but the usual
# pattern is:
#  1. For each output array axis, identify it with zero or more corresponding
#  input array axes.
#  2. If all those input axes are sharded the same as each other, shard the
#  output axis the same way; otherwise, error (and require an explicit
#  `out_sharding` argument).
#  3. After all output array axes are decided that way, if an output array
#  sharding mentions the same mesh axis more than once, error (and require an
#  explicit `out_sharding`).
#
# Here are some example rules:
# * nullary ops like `jnp.zeros`, `jnp.arange`: These ops create arrays out of whole
# cloth so they don’t have input shardings to propagate. Their output is
# unsharded by default unless overridden by the `out_sharding` kwarg.
# * unary elementwise ops like `sin`, `exp`: The output is sharded the same as
# the input.
# * binary ops (`+`, `-`, `*` etc.): Axis shardings of “zipped” dimensions must
# match (or be None). “Outer product” dimensions (dimensions that appear in only
# one argument) are sharded as they are in the input. If the result ends up
# mentioning a mesh axis more than once it’s an error.
#
# The contraction ops like `jnp.dot` and `jnp.einsum` also have some interesting
# cases. For example, the result of `jnp.dot(x: f32[8,4@X], y:f32[4@X,16])`,
# where the shared contracting axis is sharded the same way, could reasonably be:
# * `f32[8,16]` (doing an all-reduce)
# * `f32[8@X,16]` (a reduce-scatter on the first axis)
# * `f32[8,16@X]` (a reduce-scatter on the second axis)
# * `f32[8,16]{U:X}` (no communication)
# Instead of automatically choosing one, JAX errors in this case and requires an
# `out_sharding` be provided, e.g. `jnp.dot(x, y, out_sharding=jax.P('X',
# None))`:

# +
x = jax.device_put(jnp.arange(8 * 4.).reshape(8, 4), jax.P(None, 'X'))
y = jax.device_put(jnp.arange(4 * 16.).reshape(4, 16), jax.P('X', None))

try:
  jnp.dot(x, y)
except Exception as e:
  print("ERROR!")
  print(e)

# +
z = jnp.dot(x, y, out_sharding=jax.P('X', None))

print(jax.typeof(z))
# -

# But there are other `jnp.dot` cases that induce communication that JAX does
# perform automatically, like `jnp.dot(x:f32[8,4], y:f32[4@x,16])` results in an
# `f32[8,16]`, likely by doing an all-gather on `y` as in FSDP.
#
# ### With `@auto_axes` the compiler chooses shardings within the decorated function
#
# If you don't want to specify the shardings of some intermediates, and instead
# want the compiler to choose them automatically, you can use the `@auto_axes`
# decorator. Under this decorator, shardings aren't queryable using `jax.typeof`.
# More specifically, `auto_axes` switches some or all mesh axis types to `Auto`,
# and `Auto` mesh axes can't appear in array types.
#
# Decorating a function with `@auto_axes` adds an `out_sharding` argument to the
# function's signature, so the final output sharding can be set by the caller.
# Alternatively, decorating with `@auto_axes(out_sharding=...)` specifies the
# final output sharding at the function definition site.
#
# For example, when our mesh axes are `Explicit`, we can't add two arrays with
# different shardings:

# +
from jax.sharding import auto_axes, explicit_axes

x = jax.device_put(np.arange(16).reshape(4, 4), jax.P("X", None))
y = jax.device_put(np.arange(16).reshape(4, 4), jax.P(None, "X"))

try:
  x + y
except Exception as e:
  print("ERROR!")
  print(e)


# -

# If we just want to specify the sharding of the result and for the compiler to
# handle the rest, we can use `auto_axes`:

# +
@auto_axes
def add2(x, y):
  print("We're in auto-sharding mode here. This is the current mesh:\n"
        f"{jax.sharding.get_abstract_mesh()}")
  return x + y

result = add2(x, y, out_sharding=jax.P("X", None))
print(f"Result type: {jax.typeof(result)}")


# -

# So `auto_axes` lets you add an `out_sharding` argument to any composition of
# operations.
#
# An `auto_axes`-decorated function can be called when the context mesh's axis
# types are `Explicit` or `Auto`, but none can be in `Manual`. By default it
# switches all mesh axis types to `Auto`; use `axes=...` to switch only a subset.
#
# ### Concrete array shardings can mention `Auto` mesh axis
#
# The sharding of a concrete `jax.Array` can be queried via `x.sharding`.
# This can only be done at the top-level. You might expect the result to be the
# same as the sharding associated with the value’s type, `jax.typeof(x).sharding`.
# It might not be! The concrete array sharding, `x.sharding`, describes the
# sharding along both `Explicit` and `Auto` mesh axes. It’s the sharding that the
# compiler eventually chose. Whereas the type-specificed sharding,
# `jax.typeof(x).sharding`, only describes the sharding along `Explicit` mesh axes.
# The `Auto` axes are deliberately hidden from the type because they’re the purview
# of the compiler. We can think of the concrete array sharding being consistent
# with, but more specific than, the type-specified sharding. For example:

# +
def compare_shardings(x):
  print(f"=== with mesh: {jax.sharding.get_abstract_mesh()} ===")
  print(f"Concrete value sharding: {x.sharding.spec}")
  print(f"Type-specified sharding: {jax.typeof(x).sharding.spec}\n")

my_array = jnp.sin(jax.device_put(np.arange(8), jax.P("X")))
compare_shardings(my_array)

@auto_axes
def check_in_auto_context(x):
  compare_shardings(x)
  return x

check_in_auto_context(my_array, out_sharding=jax.P("X"))
# -

# Notice that at the top level, where we’re currently in a fully `Explicit` mesh context,
# the concrete array sharding and type-specified sharding agree.
#
# But under the `auto_axes` decorator we’re in a fully `Auto` mesh context and the
# two shardings disagree: the type-specified sharding is `P(None)` whereas the concrete
# array sharding is `P("X")` (though it could be anything! It’s up to the compiler).
#
# ## Auto sharding mode decides shardings automatically during compilation
#
# While the `auto_axes` decorator is useful for temporarily switching mesh axis
# types from `Explicit` to `Auto`, you can also construct a `Mesh` with `Auto`
# axis types, e.g. at the top level:

# +
Auto = jax.sharding.AxisType.Auto
auto_mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(auto_mesh)

x = jax.device_put(jnp.arange(8 * 4. ).reshape(8, 4 ), jax.P(None, 'X'))
y = jax.device_put(jnp.arange(4 * 16.).reshape(4, 16), jax.P('X', None))

z = jnp.dot(x, y)  # not an error!
# -

# Instead of getting an error, the compiler decided the sharding of the result!

print(z.sharding)  # works at the top-level only (i.e. outside `jit`)


# Whether using top-level `Auto` mesh axes, or using the `auto_axes` decorator,
# you can provide hints to the compiler about how intermediates should be sharded
# using `jax.lax.with_sharding_constraint`:

# +
@jax.jit
def f(x, y):
  z = jnp.dot(x, y)
  z = jax.lax.with_sharding_constraint(z, jax.P('X', None))
  return z

z = f(x, y)
print(z.sharding)


# -

# It's also valid to call `jax.lax.with_sharding_constraint` with `Explicit` mode
# axes; for any `Explicit` mesh axes, it acts like an assertion that the
# argument's sharding matches the specified sharding.
#
# You can locally switch mesh axis types to `Explicit` using the `@explicit_axes`
# decorator:

# +
@explicit_axes
def explicit_g(y):
  print(f'mesh inside g: {jax.sharding.get_abstract_mesh()}')
  print(f'y.sharding inside g: {jax.typeof(y) = }')
  z = y * 2
  print(f'z.sharding inside g: {jax.typeof(z) = }', end='\n\n')
  return z

@jax.jit
def f(arr1):
  print(f'mesh inside f: {jax.sharding.get_abstract_mesh()}', end='\n\n')
  x = jnp.sin(arr1)
  z = explicit_g(x, in_sharding=jax.P("X", "Y"))
  return z + 1

x = jax.device_put(np.arange(16).reshape(4, 4), jax.P("X", "Y"))
f(x)
# -

# It's a kind of dual to `auto_axes`, where you specify `in_shardings` rather
# than `out_shardings`.
#
# ## Manual mode lets you write explicit collectives with a per-device view of data
#
# Using `jax.shard_map` sets mesh axis types to `Manual`:

# +
mesh = jax.make_mesh((4, 2), ('X', 'Y'))
jax.set_mesh(mesh)

x = jax.device_put(jnp.arange(8 * 4. ).reshape(8, 4 ), jax.P(None, 'X'))
y = jax.device_put(jnp.arange(4 * 16.).reshape(4, 16), jax.P('X', None))

@jax.shard_map(out_specs=jax.P('X', None))
def matmul(x_shard, y_shard):
  z_summand = jnp.dot(x_shard, y_shard)
  return jax.lax.psum_scatter(z_summand, 'X', tiled=True)

z = matmul(x, y)
print(jax.typeof(z))

z_ref = jnp.dot(x, y, out_sharding=jax.P('X', None))
print(jnp.allclose(z_ref, z))
# -

# For details, see [the `shard_map` tutorial](https://docs.jax.dev/en/latest/notebooks/shard_map.html).
