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

The only interesting part is the inner loop. Ampere has no hardware
asynchrony for the matmul itself — `mma` reads from shared memory and runs on
the warp — so the thing we must overlap by hand is the *global → shared* copy.
That is what `emit_pipeline` does: it runs the `K//bk` steps of the reduction
as a software pipeline with `stages` buffers in flight, issuing the `cp.async`
for step `k + stages` while step `k`'s `mma` is executing. Without this the
kernel is purely memory-latency bound and lands at a small fraction of peak.

`init_carry` makes the fp32 accumulator the pipeline's carry, so it lives in
registers across all `K//bk` steps and only gets written out — cast down to
bf16 — once, at the end.

What this kernel does *not* do: it accepts whatever order the GPU happens to
schedule its blocks in. That is the next problem.

```python
M = K = N = 8192
DT = jnp.bfloat16

def matmul1(a, b, bm=128, bn=128, bk=32, stages=4):
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
        kernel_name='matmul1',
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

The fix costs one line of index arithmetic and no change to the inner loop:
split `i` into an outer band index and an inner position, and launch with the
inner axis *last* so it varies fastest. Now the resident blocks form a compact
`mt × (something)` rectangle instead of a long strip, and the `a` and `b` tiles
they need overlap heavily. 

`mt` is the knob: too small and you are back to a strip, too large and the
rectangle's working set overflows L2 again.

```python
def matmul2(a, b, bm=128, bn=128, bk=32, stages=4, mt=8):
    def body(a_gmem, b_gmem, o_gmem):
        # The only difference from matmul1: i is reassembled from the outer
        # band index and the position within the band.
        i = jax.lax.axis_index('i_outer') * mt + jax.lax.axis_index('i_inner')
        j = jax.lax.axis_index('j')

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
        grid=(M // bm // mt, N // bn, mt),
        grid_names=('i_outer', 'j', 'i_inner'),
        kernel_name='matmul2'
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
def matmul3(a, b, bm=128, bn=128, bk=32, stages=4, mt=16, occ=2):
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
        kernel_name='matmul3',
    )(a, b)
```

# Kernel 4: Persistent with Tile Pre-fetching

This one closes that seam, and it is the only kernel here that gives up
`emit_pipeline` to do it. The idea: the pipeline should not drain at a tile
boundary. While we are finishing the last few `k` steps of tile `t`, the copies
in flight should already be the *first* `k` steps of tile `t+1`.

That requires knowing the next tile's index before we finish the current one,
which is why the loop is restructured around a one-iteration lag. `nd_loop`
carries the previous index, and each iteration computes the tile it was handed
*last* time while it knows the index it was handed *this* time. The first
iteration computes nothing and only primes the buffers; one extra `tile()` call
after the loop drains the final tile, which by then has `stages - 1` steps'
worth of data already resident — hence `last=True` and the shorter `steady`
bound.

So we hand-roll the pipeline. `a_smem`/`b_smem` are explicit `stages`-slot
scratch buffers, `fetch` issues the two `cp.async`s for one `k` step into one
slot, and `k_step` waits on the right number of outstanding copies, does the
`mma`, then immediately refills the slot it just consumed. The `jax.lax.cond`
in `k_step` is the whole trick: if `k + stages` is still within this tile's `K`
range, prefetch that; otherwise prefetch step `s - nk` *of the next tile*. The
buffers never go empty.

```python
def matmul4(a, b, bm=128, bn=128, bk=32, stages=4, mt=16, occ=2,
            name='matmul4'):
    """a @ b, persistent: each walking a stride of tiles."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem):
        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def tile(ij, nxt, last=False):
            i, j = ij

            def k_step(k, acc, wait_count):
                plgpu.wait_gmem_to_smem(wait_count)
                slot = jax.lax.rem(k, stages)
                acc = plgpu.mma(acc, a_smem.at[slot][...],
                                b_smem.at[slot][...])
                s = k + stages
                if last:
                    jax.lax.cond(s < nk, lambda: fetch(i, j, s, slot), lambda: None)
                else:
                    jax.lax.cond(s < nk,
                                 lambda: fetch(i, j, s, slot),
                                 lambda: fetch(nxt[0], nxt[1], s - nk, slot))
                return acc

            zero = jnp.zeros((bm, bn), jnp.float32)
            steady = nk - (stages - 1) if last else nk
            acc = jax.lax.fori_loop(
                0, steady, lambda k, acc: k_step(k, acc, (stages - 1) * 2),
                zero)
            for k in range(steady, nk):
                acc = k_step(jnp.int32(k), acc, (nk - 1 - k) * 2)
            o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

        # The carry is the previous iteration's index
        @plgpu.nd_loop((ni, nj), collective_axes='sm', tiling=(mt, 1),
                       init_carry=(jnp.int32(0), jnp.int32(0)))
        def last_ij(loop_info, carry):
            ij = loop_info.index

            # Only the very first tile has to prime the pipeline itself.
            @pl.when(loop_info.local_index == 0)
            def _():
                for k in range(stages):
                    fetch(ij[0], ij[1], jnp.int32(k), k)

            @pl.when(loop_info.local_index > 0)
            def _():
                tile(carry, ij)

            return ij

        # The loop fetched one tile further than it computed.
        tile(last_ij, last_ij, last=True)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)],
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name=name,
        compiler_params=plgpu.CompilerParams(),
    )(a, b)
```

# Kernel 5: Square Warp Tile Shape

Kernel 4 implemented persistent scheduling with tile pre-fetching and 4 stages.
However, it left one critical hardware knob at its default: how the 128×128 CTA
tile is partitioned among the CTA's 4 warps.

By default, Mosaic picks `m_warps=4` (and `n_warps=1`). Each warp gets a tall,
thin 32×128 rectangular band of the output tile. Why is that suboptimal?

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

Compare the two layouts per warp per $k$ step (in `LDSM.x4` instructions):
- **32×128 warp tile (`m_warps=4`):** Loads $32 \times 32$ of $A$ (4 loads) and
  $32 \times 128$ of $B$ (16 loads) = **20 `LDSM` instructions**.
- **64×64 warp tile (`m_warps=2`):** Loads $64 \times 32$ of $A$ (8 loads) and
  $32 \times 64$ of $B$ (8 loads) = **16 `LDSM` instructions**.

Moving to a square tile eliminates **20% of shared-memory load instructions** per
$k$ step (16 instead of 20). It also significantly eases register pressure: at
32×128, ptxas allocates 240 registers per thread, leaving just 15 spare registers
before spilling. At 64×64, register usage drops to 226 registers per thread
(freeing 14 registers).

So we want `m_warps=2`. There is no parameter for it: the width is carried by
the operands' layouts, and `plgpu.Layout.MMA_LHS`/`MMA_RHS`/`MMA_ACC` name them
for any width. Casting the operands is how you pick one.

```python
lhs = plgpu.Layout.MMA_LHS(DT, m_warps=2)       # A, 64×32 per warp
rhs = plgpu.Layout.MMA_RHS(DT, m_warps=2)       # B, 32×64 per warp
acc_l = plgpu.Layout.MMA_ACC(jnp.float32, m_warps=2)

acc = plgpu.mma(acc,
                plgpu.layout_cast(a_smem.at[slot][...], lhs),
                plgpu.layout_cast(b_smem.at[slot][...], rhs))
```

The casts go *inside* the loop. That is the whole kernel: kernel 4 with three layouts named at the top and two casts in `k_step`.

```python
def matmul5(a, b, bm=128, bn=128, bk=32, stages=4, mt=16, occ=2,
            m_warps=2, name='matmul5'):
    """matmul4, with the warp split asked for rather than inferred."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)

    lhs = plgpu.Layout.MMA_LHS(DT, m_warps=m_warps)
    rhs = plgpu.Layout.MMA_RHS(DT, m_warps=m_warps)
    acc_l = plgpu.Layout.MMA_ACC(jnp.float32, m_warps=m_warps)

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem):
        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def tile(ij, nxt, last=False):
            i, j = ij

            def k_step(k, acc, wait_count):
                plgpu.wait_gmem_to_smem(wait_count)
                slot = jax.lax.rem(k, stages)
                # The two casts are the only difference from kernel 4.
                acc = plgpu.mma(
                    acc,
                    plgpu.layout_cast(a_smem.at[slot][...], lhs),
                    plgpu.layout_cast(b_smem.at[slot][...], rhs))
                s = k + stages
                if last:
                    jax.lax.cond(s < nk, lambda: fetch(i, j, s, slot), lambda: None)
                else:
                    jax.lax.cond(s < nk,
                                 lambda: fetch(i, j, s, slot),
                                 lambda: fetch(nxt[0], nxt[1], s - nk, slot))
                return acc

            # ...and the cast on the zero, so the carry starts in the right layout.
            zero = plgpu.layout_cast(jnp.zeros((bm, bn), jnp.float32), acc_l)
            steady = nk - (stages - 1) if last else nk
            acc = jax.lax.fori_loop(
                0, steady, lambda k, acc: k_step(k, acc, (stages - 1) * 2),
                zero)
            for k in range(steady, nk):
                acc = k_step(jnp.int32(k), acc, (nk - 1 - k) * 2)
            o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

        @plgpu.nd_loop((ni, nj), collective_axes='sm', tiling=(mt, 1),
                       init_carry=(jnp.int32(0), jnp.int32(0)))
        def last_ij(loop_info, carry):
            ij = loop_info.index

            @pl.when(loop_info.local_index == 0)
            def _():
                for k in range(stages):
                    fetch(ij[0], ij[1], jnp.int32(k), k)

            @pl.when(loop_info.local_index > 0)
            def _():
                tile(carry, ij)

            return ij

        tile(last_ij, last_ij, last=True)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)],
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name=name,
    )(a, b)
```

# Kernel 6: Pace the copies by splitting one of them

Kernel 5 exhausts the obvious knobs, and the profile says the machine is idle
on average but contended in bursts: `mio_throttle` and `math_pipe_throttle`
dominate, ~80% of cycles have zero eligible warps, and neither pipe is near its
roof. The warps are arriving at the same resource at the same moment.

One `k` step issues eight `LDGSTS.128` — 8 KB each of `A` and `B`, 64 bytes per
thread per matrix — from all four warps at once. But not as one burst: Mosaic
emits a barrier *before* each `async_copy`, so the barrier between the `A` copy
and the `B` copy already splits them into two bursts of four. That barrier is
not overhead. It is the only thing pacing the memory pipe.

So split further. Slicing `A`'s copy into `asplit` pieces makes Mosaic emit
`asplit` barriers where it emitted one, with no new primitive:

```
asplit=1   3 barriers per k step   LDGSTS bursts of 4, 4
asplit=2   4 barriers per k step   bursts of 2, 2, 4
asplit=4   6 barriers per k step   bursts of 1, 1, 1, 1, 4
```

Where you may cut is fixed by two rules. The tiled copy path wants row tiles
divisible by 4: `A` is `(128, 32)`, so cutting along `m` gives 8 or 4 row tiles
and is fine, while `B` cut along `k` gives `(16, 128)` — two row tiles — and is
rejected. And `B`'s 128 columns are one 128-byte swizzle period, so cutting
along `n` would slice inside it. `B` stays whole.

One bookkeeping consequence: a `fetch` now issues `asplit + 1` copies, so every
`wait_gmem_to_smem` count scales with it. Derive them from that number, never
from the constant `2` — a count that is too small does not raise, it lets the
`mma` read a slot that has not landed.

Everything else is kernel 5 unchanged, so the diff is `fetch` and the two wait
counts:

```python
def matmul6(a, b, bm=128, bn=128, bk=32, stages=4, mt=16, occ=2,
            m_warps=2, asplit=2, name='matmul6'):
    assert bm % (32 * asplit) == 0, 'each A slice needs >= 4 row tiles'
    per_fetch = asplit + 1          # copies issued by one fetch

    def fetch(i, j, k, slot):
        step = bm // asplit
        for p in range(asplit):     # asplit barriers, one before each copy
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm + p * step, step), pl.ds(k * bk, bk)],
                a_smem.at[slot, pl.ds(p * step, step)],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
        plgpu.copy_gmem_to_smem(     # B is uncuttable; it stays one burst
            b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
            b_smem.at[slot],
            oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

    acc = jax.lax.fori_loop(
        0, steady,
        lambda k, acc: k_step(k, acc, (stages - 1) * per_fetch), zero)
    for k in range(steady, nk):
        acc = k_step(jnp.int32(k), acc, (nk - 1 - k) * per_fetch)
```
