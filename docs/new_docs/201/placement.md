---
jupytext:
  formats: md:myst
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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(jax-201-placement)=
# Data placement

<!--* freshness: { reviewed: '2026-07-10' } *-->

Every array lives somewhere: in the memory of one device, or spread across
the memories of many. The unit of placement in JAX is the **mesh** — a set of
devices arranged in a grid with named axes. A large program might place
different arrays on different meshes and move data between them; the humble
everyday case, an array sitting on one device, is just placement on a
single-device mesh. This page covers how placement is decided by default, how
to control it with {func}`jax.device_put`, what it means for an array to be
*committed* to a mesh, and how to move data between meshes.

(The full story of how data is laid out *across* a mesh's devices — shardings,
partition specs, and parallelism — is the next page, {doc}`sharding`. Here we
only need the idea of a mesh itself.)

We'll simulate eight devices on CPU:

```{code-cell}
import jax

jax.config.update('jax_num_cpu_devices', 8)

import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
```

A mesh over all eight devices, arranged along one named axis:

```{code-cell}
full = jax.make_mesh((8,), ('x',))
full
```

And, using {func}`jax.make_mesh`'s `devices` argument, some smaller meshes
over subsets of the same hardware — two halves, and a couple of single-device
meshes:

```{code-cell}
mesh_a = jax.make_mesh((4,), ('x',), devices=full.devices[:4])
mesh_b = jax.make_mesh((4,), ('x',), devices=full.devices[4:])

m0 = jax.make_mesh((1,), ('x',), devices=full.devices[:1])
m1 = jax.make_mesh((1,), ('x',), devices=full.devices[1:2])
```

## Default placement, and commitment

When you create an array without saying where it should go, JAX places it
somewhere sensible — and leaves it *uncommitted*:

```{code-cell}
x = jnp.arange(4.0)
print(x.committed)
print(x.sharding)
```

Uncommitted means the array isn't attached to its location: it's free to
follow. If it's used in a computation together with data that *is* committed
somewhere, JAX will move it there implicitly.

To place an array deliberately, use {func}`jax.device_put` with a sharding —
a mesh plus a description of how the array is laid out over it:

```{code-cell}
data = np.arange(8.0)
xa = jax.device_put(data, NamedSharding(mesh_a, jax.P('x')))
print(xa.committed)
print(jax.typeof(xa))
```

Now the array is **committed** to `mesh_a` — placement is part of its JAX
type (the `@x` in `float32[8@x]`), and it stays put. Computations follow
their data: an operation on `xa` runs on `mesh_a`'s devices.

Commitment is a promise JAX enforces. Combining arrays committed to
*different* meshes in one operation is an error, not an implicit transfer:

```{code-cell}
:tags: [raises-exception]

a = jax.device_put(1.0, NamedSharding(m0, jax.P()))
b = jax.device_put(2.0, NamedSharding(m1, jax.P()))

a + b  # committed to different meshes!
```

This is deliberate: transfers are expensive, so JAX makes you say where data
should move rather than shuffling it behind your back. Uncommitted values,
by contrast, follow along freely:

```{code-cell}
c = jnp.float32(3.0)   # uncommitted
print((a + c).sharding)  # runs on m0, where `a` is committed
```

(In older code you'll see `jax.device_put(x, some_device)`, committing an
array to a single device. That's this same model: think of it as committing
to a single-device mesh.)

## Moving data between meshes

The same function that places data also moves it: calling
{func}`jax.device_put` on an already-placed array with a sharding on a
different mesh transfers it there.

```{code-cell}
xb = jax.device_put(xa, NamedSharding(mesh_b, jax.P('x')))
print(xb.sharding.mesh == mesh_b)
```

Two related operations are worth distinguishing:

- **Changing layout within a mesh** — same devices, different partitioning —
  is *resharding*, covered in {doc}`sharding` (see `jax.reshard`, and
  `device_put` with a new spec on the same mesh).
- **Changing meshes** — moving data to a different set of devices — is this
  page's operation, and `jax.device_put` is the tool.

One subtle case worth calling out: a mesh isn't just a *set* of devices, it's
an *arrangement* of them. So keeping an array's data on the same overall set
of devices but changing the device *order* in the sharding is also a change
of mesh — which makes it a `device_put`, not a reshard. And that's not
bureaucracy: the same partition spec over differently-ordered meshes assigns
different data to each device, so real data movement is required.

```{code-cell}
mesh_rev = jax.make_mesh((8,), ('x',), devices=full.devices[::-1])

x = jax.device_put(np.arange(8.0), NamedSharding(full, jax.P('x')))
y = jax.device_put(x, NamedSharding(mesh_rev, jax.P('x')))  # same devices, new order

def data_by_device(arr):
  return [float(s.data[0]) for s in sorted(arr.addressable_shards,
                                           key=lambda s: s.device.id)]

print(data_by_device(x))  # what each of devices 0, 1, ..., 7 holds
print(data_by_device(y))
```

As mathematical values, `x` and `y` are identical — but every element has
moved: device 0 held `0.0` before and holds `7.0` after. (And since `x` and
`y` are committed to different meshes, combining them in one operation would
be an error, exactly as above.)

**Mesh changes are runtime-level operations.** Everything in this section
happens at the top level of your program — and only there. A compiled
computation belongs to one fixed mesh: under `jax.jit` (or inside any
staged-out control flow like `jax.lax.scan`), you can change an array's
*layout* on the mesh, but you cannot move data to a *different* mesh.
Attempting a cross-mesh `device_put` inside `jit` is an error:

```{code-cell}
:tags: [raises-exception]

@jax.jit
def f(x):
  return jax.device_put(x, NamedSharding(mesh_b, jax.P('x')))

f(xa)  # error: can't move to a different mesh inside jit!
```

Moving between meshes is the runtime's job, done from Python between compiled
computations — which is why the pipeline example below drives each
stage-to-stage transfer from the top level.

Like everything else in JAX's runtime, `device_put` is asynchronous
({ref}`jax-201-async-dispatch`): it returns immediately, and the transfer
proceeds concurrently with whatever Python does next. This has a powerful
consequence: computations and transfers on *different* meshes run in
parallel whenever their inputs are ready, because each device works through
its own queue independently. That's enough to express pipeline parallelism
at the runtime level, with no compiler involvement — stages committed to
disjoint meshes, `device_put` moving each microbatch from stage to stage:

```{code-cell}
stage_meshes = [jax.make_mesh((2,), ('x',), devices=full.devices[2*i:2*i+2])
                for i in range(4)]

@jax.jit
def stage(x):
  return jnp.sin(x) + 1.0   # stand-in for real per-stage computation

microbatches = [np.full((256, 256), float(i)) for i in range(8)]

results = []
for mb in microbatches:
  for m in stage_meshes:
    mb = jax.device_put(mb, NamedSharding(m, jax.P('x')))
    mb = stage(mb)
  results.append(mb)

print(results[-1].sharding.mesh == stage_meshes[-1])
```

Each microbatch is enqueued through the stages sequentially, but the queues
are independent: while stage 1 works on microbatch 0, stage 0 has already
started microbatch 1, with the transfers overlapping the computation. (For a
richer treatment of pipelining, see the microbatched gradient-accumulation
example in {doc}`/array_refs`.)

Everything on this page generalizes to multiple processes: meshes can span
hosts, can cover just a subset of the devices in a cluster, and
`jax.device_put` can transfer between meshes across processes over
high-speed interconnects. That story belongs to the multi-process systems
docs — see {doc}`/multi_process` in the meantime.

## Notes

- `jax.device_put(x)` with no target is (at most) a transfer to the default
  placement, and the result is uncommitted.
- `jax.device_put` accepts `donate=True` to reuse the input's buffers when
  possible, in the same spirit as `jit`'s buffer donation
  ({ref}`jax-201-buffer-donation`).
- Placement round-trips through the host too: `np.asarray(x)` copies device
  data back to host memory as a NumPy array.

## Next steps

With placement in hand, the natural question is what happens *within* a
mesh: how an array's data is partitioned over the mesh's devices, and how
JAX turns one program into coordinated per-device computation. That's
{doc}`sharding`.
