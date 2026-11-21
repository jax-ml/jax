```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
```

This is a walkthrough of six Pallas kernels for `a @ b` on an Ampere-class GPU
(A100), each one fixing the bottleneck the previous one hit. 

# Kernel 1: Basic Ampere Matmul

In our baseline kernel, we'll implement the obvious path: partition the output
into `bm × bn` tiles, give one tile to
each CTA, and have that CTA walk the whole `K` dimension accumulating into
registers. The grid is `(M//bm, N//bn)`, so each program knows which output
tile it owns from its two axis indices `i` and `j`.

The only interesting part is the inner loop: we'll pipeline the *global → shared* copy.
That is what `emit_pipeline` does: it runs the `K//bk` steps of the reduction
as a software pipeline with `stages` buffers in flight, issuing the `cp.async`
for step `k + stages` while step `k`'s `mma` is executing. Without this the
kernel is purely memory-latency bound and lands at a small fraction of peak.

`init_carry` makes the fp32 accumulator the pipeline's carry, so it lives in
registers across all `K//bk` steps and only gets written out — cast down to
bf16 — once, at the end. By itself, this basic kernel already gets us 91.6% of the
performance of CuBLAS. 

What this kernel does *not* do: it accepts whatever order the GPU happens to
schedule its blocks in. That is the next problem.

```python
M = K = N = 8192
DT = jnp.bfloat16

def kernel1(a, b, bm=128, bn=128, bk=32, stages=4):
    def body(a_gmem, b_gmem, o_gmem):
        i, j = (jax.lax.axis_index(n) for n in 'ij')
        def step(indices, a_smem, b_smem, acc):
            return plgpu.mma(acc, a_smem[...], b_smem[...])

        acc = plgpu.emit_pipeline(
            step,
            grid=(K // bk,),
            in_specs=[
                plgpu.BlockSpec((bm, bk), lambda k: (i, k)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j)),
            ],
            max_concurrent_steps=stages,
            init_carry=jnp.zeros((bm, bn), jnp.float32),
        )(a_gmem, b_gmem)

        o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(M // bm, N // bn),
        grid_names=('i', 'j'),
        kernel_name='kernel1',
    )(a, b)
```

# Kernel 2: Tile the grid for L2 locality

Kernel 1 reads each tile of `a` once per column of the output and each tile of
`b` once per row — far more traffic than the matrices themselves. Whether that
traffic actually reaches HBM depends on whether the concurrent blocks are
*reusing each other's* reads out of L2.

The default launch order is roughly row-major in the grid, so the ~108·k blocks
resident at any moment sweep across a very wide band of `j` while sharing only a
handful of `i` rows. The working set is the whole width of `b`, which does not
fit in L2, so almost nothing hits.

The fix costs one line and no change to the inner loop: keep a flat 1-D grid
of `ni * nj` blocks and let `plgpu.planar_snake` map the linear block index to
`(i, j)` along a locality-friendly curve. With `minor_dim=0, tile_width=mt` it
walks `mt` rows of `i` before stepping in `j`, so the resident blocks form a
compact `mt × (something)` rectangle instead of a long strip, and the `a` and
`b` tiles they need overlap heavily. It also serpentines: consecutive bands run
in opposite directions in `j`, so the band seam reuses the `b` tiles still warm
in L2 — though at these shapes a band is 1024 tiles wide and only a few waves
ever straddle one, so do not expect to measure it. This change gets us to 94.1%
of the speed of CuBLAS. 

`mt` is the knob: too small and you are back to a strip, too large and the
rectangle's working set overflows L2 again.

```python
def kernel2(a, b, bm=128, bn=128, bk=32, stages=4, mt=8):
    ni, nj = M // bm, N // bn

    def body(a_gmem, b_gmem, o_gmem):
        # The only difference from kernel1: the grid is flat and the curve,
        # not the launch order, decides which tile this block owns.
        i, j = plgpu.planar_snake(jax.lax.axis_index('t'), (ni, nj), 0, mt)

        def step(indices, a_smem, b_smem, acc):
            with jax.named_scope("mma"):
                return plgpu.mma(acc, a_smem[...], b_smem[...])

        acc = plgpu.emit_pipeline(
            step,
            grid=(K // bk,),
            in_specs=[
                plgpu.BlockSpec((bm, bk), lambda k: (i, k)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j)),
            ],
            max_concurrent_steps=stages,
            init_carry=jnp.zeros((bm, bn), jnp.float32),
        )(a_gmem, b_gmem)

        o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(ni * nj,),
        grid_names=('t',),
        kernel_name='kernel2'
    )(a, b)
```

# Kernel 3: Persistent

At 8192³ with 128×128 tiles there are 4096 output tiles and only ~108 SMs, so
the hardware runs about 38 waves of blocks. Every wave boundary costs
something: blocks launch, prime their `stages`-deep pipeline from cold shared
memory, do `K//bk` steps of work, drain, and die. The pipeline prologue is pure
latency, paid 4096 times, and the tail of each wave leaves SMs idle while
stragglers finish.

A persistent kernel inverts the mapping. Launch exactly as many CTAs as can be
resident — `occ` blocks per SM — and let each one loop over a strided subset of
the tile space itself. `nd_loop` does the bookkeeping: with
`collective_axes='sm'` it hands CTA `p` the tiles `p`, `p + num_sms`, … of the
`(ni, nj)` space, and `tiling=(mt, 1)` makes that walk take `mt` steps in `i`
before moving in `j`, which is the same L2 locality kernel 2.

The body inside the loop is unchanged from kernel 2. What we have bought is
that scheduling overhead is now paid once per *SM* instead of once per tile.
What we have *not* bought yet is anything at the seam between consecutive
tiles: each tile still cold-starts its own `emit_pipeline`, so between the last
`mma` of one tile and the first of the next, the tensor cores sit idle for a
full global-memory round trip.

```python
def kernel3(a, b, bm=128, bn=128, bk=32, stages=4, mt=8, occ=2):
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)

    def body(a_gmem, b_gmem, o_gmem):
        @plgpu.nd_loop((ni, nj), collective_axes='sm', tiling=(mt, 1))
        def _(loop_info):
            i, j = loop_info.index

            def step(indices, a_smem, b_smem, acc):
                with jax.named_scope("mma"):
                    return plgpu.mma(acc, a_smem[...], b_smem[...])

            acc = plgpu.emit_pipeline(
                step,
                grid=(nk,),
                in_specs=[
                    plgpu.BlockSpec((bm, bk), lambda k: (i, k)),
                    plgpu.BlockSpec((bk, bn), lambda k: (k, j)),
                ],
                max_concurrent_steps=stages,
                init_carry=jnp.zeros((bm, bn), jnp.float32),
            )(a_gmem, b_gmem)

            o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name='kernel3',
    )(a, b)
```

# Kernel 4: Persistent with Tile Pre-fetching

This one closes that seam, and it is the only kernel here that gives up
`emit_pipeline` to do it. The idea: the pipeline should not drain at a tile
boundary. While we are finishing the last few `k` steps of tile `t`, the copies
in flight should already be the *first* `k` steps of tile `t+1`.

The whole trick is to compute the tile index instead of being handed it. This
CTA's `t`'th tile is grid tile `p + t * num_sms`, so the *next* tile's index is
just `snake(t + 1)` — available whenever we want it, with nothing carried
between iterations. `planar_snake` maps it the same way kernel 2 did, so the
`mt`-band L2 locality comes along for free. The kernel is then two pieces:

1. **Prime** — fetch the first `stages` `k` steps of this CTA's first tile.
2. **The k loop** — wait, `mma` on slot `k % stages`, then refill *that same
   slot* with step `k + stages`, which a `cond` redirects onto the next tile
   once `k` passes `nk - stages`. The last tile clamps `nxt` to itself, so its
   final fetches land in slots nobody reads again; no special case for it.

`nk % stages == 0` is what lets `slot = k % stages` line up at the tile
boundary; at `nk=256, stages=4` it holds. Per-SM tile counts differ by one
(4096 tiles over 216 CTAs is 18 or 19), hence the `my_tiles` ceiling division.

```python
def kernel4(a, b, bm=128, bn=128, bk=32, stages=4, mt=8, occ=2):
    """a @ b, persistent: the pipeline runs straight through tile boundaries."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)
    assert nk % stages == 0

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem):
        p = jax.lax.axis_index('sm')
        my_tiles = (tiles - p - 1) // num_sms + 1   # ceil

        def snake(t):
            """(i, j) of this CTA's t'th tile."""
            return plgpu.planar_snake(p + t * num_sms, (ni, nj), 0, mt)

        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def k_step(k, acc, ij, nxt):
            plgpu.wait_gmem_to_smem((stages - 1) * 2)
            slot = jax.lax.rem(k, stages)
            acc = plgpu.mma(acc, a_smem.at[slot][...], b_smem.at[slot][...])
            # Refill the slot just drained, with the step `stages` ahead --
            # which belongs to the next tile once k passes nk - stages.
            s = k + stages
            jax.lax.cond(s < nk,
                         lambda: fetch(ij[0], ij[1], s, slot),
                         lambda: fetch(nxt[0], nxt[1], s - nk, slot))
            return acc

        def tile(t, _):
            ij = snake(t)
            # Clamped, so the last tile prefetches itself instead of
            # running off the end.
            nxt = snake(jnp.minimum(t + 1, my_tiles - 1))

            acc = jax.lax.fori_loop(
                0, nk, lambda k, acc: k_step(k, acc, ij, nxt),
                jnp.zeros((bm, bn), jnp.float32))

            o_gmem[pl.ds(ij[0] * bm, bm),
                   pl.ds(ij[1] * bn, bn)] = acc.astype(DT)
            return _

        # Prime the pipeline, then walk this CTA's tiles.
        i0, j0 = snake(jnp.int32(0))
        for k in range(stages):
            fetch(i0, j0, jnp.int32(k), k)
        jax.lax.fori_loop(0, my_tiles, tile, None)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)],
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name='kernel4',
        compiler_params=plgpu.CompilerParams(),
    )(a, b)
```

# Kernel 5: Square Warp Tile Shape

Kernel 4 implemented persistent scheduling with tile pre-fetching and 4 stages.
However, it left one critical hardware knob at its default: how the 128×128 CTA
tile is partitioned among the CTA's 4 warps.

`m_warps` is how many of the warpgroup's 4 warps are laid along the `M`
dimension of the accumulator; the other factor, `n_warps = 4 // m_warps`, goes
along `N`. Only 1, 2 and 4 are legal, so the three choices are 4×1, 2×2 and
1×4, and at a 128×128 tile they hand each warp:

| `m_warps` | `n_warps` | warp's slab of the output | A it must read | B it must read |
|---|---|---|---|---|
| **4** (the default) | 1 | 32×128 | rows 32·w … 32·w+31 | **all 128 columns** |
| 2 | 2 | 64×64 | 64 rows | 64 columns |
| 1 | 4 | 128×32 | **all 128 rows** | 32 columns |

Mosaic picks 4 by default. Why is that suboptimal here?

Registers are private to each warp, and `LDSM` (load shared matrix) is a per-warp
instruction. If two warps need the same elements from shared memory, each warp
must issue its own separate `LDSM`. A warp computing an $R \times C$ slab of the
output must load $R \cdot bk$ elements of $A$ and $bk \cdot C$ elements of $B$
for each $k$ step, producing $R \cdot C \cdot bk$ MACs.

The arithmetic intensity against shared memory scales with the perimeter-to-area
ratio:
- Math scales as the area: $bk \cdot R \cdot C$
- Shared memory reads scale as the perimeter: $bk \cdot (R + C)$

For a fixed area ($R \cdot C = 128 \times 128 / 4 = 4096$), the perimeter $R + C$
is strictly minimized when the tile is a square: $R = C = 64$!

Concretely, per warp per `k` step at `bk=32`:

| `m_warps` | A elements | B elements | total | MACs |
|---|---|---|---|---|
| 4 | 32·32 = 1024 | 32·128 = 4096 | **5120** | 131072 |
| **2** | 64·32 = 2048 | 32·64 = 2048 | **4096** | 131072 |

Same math, 20% fewer elements through `LDSM`.

So we want `m_warps=2`, the square 64×64. There is no parameter for it: the
split is carried by the operands' layouts, and
`plgpu.Layout.MMA_LHS`/`MMA_RHS`/`MMA_ACC` name them for any of the three.
Casting the operands is how you pick one — `mma` reads the split back off
whichever operand was cast and constrains the rest to match.

```python
lhs = plgpu.Layout.MMA_LHS(DT, m_warps=2)       # A, 64×32 per warp
rhs = plgpu.Layout.MMA_RHS(DT, m_warps=2)       # B, 32×64 per warp
acc_l = plgpu.Layout.MMA_ACC(jnp.float32, m_warps=2)

acc = plgpu.mma(acc,
                plgpu.layout_cast(a_smem.at[slot][...], lhs),
                plgpu.layout_cast(b_smem.at[slot][...], rhs))
```

The casts go *inside* the loop. That is the whole kernel: kernel 4 with three layouts named at the top and two casts in `k_step`. This gets us to 94.4% of CuBLAS performance. 

```python
def kernel5(a, b, bm=128, bn=128, bk=32, stages=4, mt=8, occ=2,
            m_warps=2):
    """kernel4, with the warp split asked for rather than inferred."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)
    assert nk % stages == 0

    lhs = plgpu.Layout.MMA_LHS(DT, m_warps=m_warps)
    rhs = plgpu.Layout.MMA_RHS(DT, m_warps=m_warps)
    acc_l = plgpu.Layout.MMA_ACC(jnp.float32, m_warps=m_warps)
    # The cast on the zero too, so the carry starts in the right layout.
    zero = plgpu.layout_cast(jnp.zeros((bm, bn), jnp.float32), acc_l)

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem):
        p = jax.lax.axis_index('sm')
        my_tiles = (tiles - p - 1) // num_sms + 1   # ceil

        def snake(t):
            """(i, j) of this CTA's t'th tile."""
            return plgpu.planar_snake(p + t * num_sms, (ni, nj), 0, mt)

        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def k_step(k, acc, ij, nxt):
            plgpu.wait_gmem_to_smem((stages - 1) * 2)
            slot = jax.lax.rem(k, stages)
            acc = plgpu.mma(
                acc,
                plgpu.layout_cast(a_smem.at[slot][...], lhs),
                plgpu.layout_cast(b_smem.at[slot][...], rhs))
            # Refill the slot just drained, with the step `stages` ahead --
            # which belongs to the next tile once k passes nk - stages.
            s = k + stages
            jax.lax.cond(s < nk,
                         lambda: fetch(ij[0], ij[1], s, slot),
                         lambda: fetch(nxt[0], nxt[1], s - nk, slot))
            return acc

        def tile(t, _):
            ij = snake(t)
            # Clamped, so the last tile prefetches itself instead of
            # running off the end.
            nxt = snake(jnp.minimum(t + 1, my_tiles - 1))

            acc = jax.lax.fori_loop(
                0, nk, lambda k, acc: k_step(k, acc, ij, nxt),
                zero)

            o_gmem[pl.ds(ij[0] * bm, bm),
                   pl.ds(ij[1] * bn, bn)] = acc.astype(DT)
            return _

        # Prime the pipeline, then walk this CTA's tiles.
        i0, j0 = snake(jnp.int32(0))
        for k in range(stages):
            fetch(i0, j0, jnp.int32(k), k)
        jax.lax.fori_loop(0, my_tiles, tile, None)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)],
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name='kernel5',
    )(a, b)
```

# Kernel 6: Stage the Epilogue Through Shared Memory

One line has survived every kernel so far, give or take what the tile index is
called:

```python
o_gmem[pl.ds(ij[0] * bm, bm), pl.ds(ij[1] * bn, bn)] = acc.astype(DT)
```

Writing the output directly from registers to global memory is our final bottleneck.

## The Bottleneck: Narrow Epilogue Stores

The accumulator registers hold data in `MMA_ACC` layout—the layout produced by tensor core matrix instructions (`HMMA.16816`).

In this layout, each lane holds scattered fragments of just 2 `bfloat16` elements (4 bytes) per row. Writing directly to global memory forces each lane to emit narrow 4-byte stores (`ST.E`). Across a full tile, this produces **64 narrow store instructions** per fragment that the memory coalescer must scramble to pack into global memory transactions.

## The Fix: The Shared Memory Bounce

On Ampere, the fastest way to write to global memory is using wide, 128-bit vectorized stores (**`ST.E.128`**), where each thread writes 16 contiguous bytes (8 `bfloat16` elements) at once. This cuts the store instructions by 4×—from 64 down to **16 `ST.E.128`** stores.

Because tensor core registers cannot be shuffled directly into 128-bit global memory vectors, we "bounce" the data through shared memory to reorder it:

1. **Write to SMEM:** Write the accumulator registers into an intermediate shared memory staging buffer (`o_smem`).
2. **Read from SMEM:** Read the data back using `plgpu.Layout.SMEM_GMEM_COPY`, which places 16 contiguous bytes in each lane.
3. **Store to GMEM:** Write out wide 128-bit stores (`ST.E.128`) to global memory.

To prevent shared memory bank conflicts during this write and readback, `o_smem` is allocated with a swizzled layout found via `plgpu.find_swizzle`.

## Preserving Occupancy: Chunking the Epilogue (`en`)

Why not stage the entire 128×128 output tile at once?

A full 128×128 tile of `bfloat16` requires 32 KB of shared memory. But shared memory is our binding resource:
- At `stages=4`, the input pipeline already uses 64 KB (32 KB for $A$ and 32 KB for $B$).
- An A100 SM has 164 KB of user shared memory.
- To maintain an occupancy of 2 CTAs per SM, each CTA cannot exceed 82 KB ($164 / 2$).

Staging all 32 KB at once would push CTA usage to $64 + 32 = 96\text{ KB}$, dropping occupancy to **1 CTA per SM** ($2 \times 96\text{ KB} = 192\text{ KB} > 164\text{ KB}$).

The solution is to chunk the epilogue across columns using the knob `en=64`:

| `en` | Staging Buffer | SMEM per CTA | CTAs / SM | Total SMEM / SM |
|---|---|---|---|---|
| None (direct) | 0 KB | 64 KB | 2 | 128 KB of 164 KB |
| **64 (chunked)** | **16 KB** | **80 KB** | **2** | **160 KB of 164 KB** |
| 128 (full tile) | 32 KB | 96 KB | 1 | 96 KB (192 > 164 KB) |

With `en=64`, the staging buffer needs only 16 KB ($128 \times 64 \times 2\text{ bytes}$), keeping total shared memory at 80 KB per CTA and fitting 2 CTAs with 4 KB to spare.

In the kernel, we loop `bn // en` times (2 iterations), staging and writing 64 columns per pass:

```python
# The epilogue: en columns at a time through SMEM
for c in range(bn // en):
    o_smem[...] = acc[:, c * en:(c + 1) * en].astype(DT)
    o_gmem[pl.ds(ij[0] * bm, bm),
           pl.ds(ij[1] * bn + c * en, en)] = plgpu.layout_cast(
        o_smem[...],
        plgpu.Layout.SMEM_GMEM_COPY((bm, en), DT, swizzle=swizzle))
```

This change gets us to 97.3% of CuBLAS performance. 

# Kernel 7: Unroll the k Loop to Fold the Slot Index

Kernel 6's mainloop issues 158 SASS instructions per k step to do 64 `HMMA`:
2.47 instructions per unit of tensor-core work. Some of those instructions do
nothing but recompute where in the shared-memory ring this step's operands
live.

## The Bottleneck: A Slot Index Computed at Runtime

Every k step selects its pipeline buffer with

```python
slot = jax.lax.rem(k, stages)
```

`k` is the `fori_loop` index, so the modulo and every SMEM address derived from
it are evaluated on the GPU, once per step, forever.

## The Fix: Unroll by `stages`

Place `stages` consecutive k steps in one loop body. Each step in that body now
has a *fixed* position in the ring — step 0 always touches slot 0, step 1 slot
1, and so on — so the slot index becomes a Python integer and the compiler
folds it away. Per four steps the SASS loses 26 `UIADD3`, 16 `LEA`, 10 `MOV`,
6 `UISETP`, 6 `PLOP3` and 4 `ULOP3`.

Each step keeps its own wait → `mma` → fetch, exactly as kernel 6 had it, so
the copy ordering and the barrier count are bit-for-bit unchanged. The only
thing that changes is what the compiler knows about `slot`. This change gets us to
97.7% of the speed of CuBLAS. 

```python
def k_step(k, acc, ij, nxt, slot):
    plgpu.wait_gmem_to_smem((stages - 1) * 2)
    acc = plgpu.mma(
        acc,
        plgpu.layout_cast(a_smem.at[slot][...], lhs),
        plgpu.layout_cast(b_smem.at[slot][...], rhs))
    s = k + stages
    jax.lax.cond(s < nk,
                 lambda: fetch(ij[0], ij[1], s, slot),
                 lambda: fetch(nxt[0], nxt[1], s - nk, slot))
    return acc

def unrolled(g, acc, ij, nxt):
    """`stages` consecutive k steps, bodies unchanged."""
    for u in range(stages):
        acc = k_step(g * stages + u, acc, ij, nxt, u)
    return acc

acc = jax.lax.fori_loop(
    0, nk // stages, lambda g, acc: unrolled(g, acc, ij, nxt), zero)
```
