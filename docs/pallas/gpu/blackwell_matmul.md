# Writing high-performance matrix multiplication kernels for Blackwell

In this guide, we'll progressively iterate on a matrix multiplication kernel.
The first implementation will be very simple, but also quite slow.
However, in just a few simple steps it can be modified into a state-of-the-art
kernel, matching or exceeding highly optimized implementations such as cuBLAS
and CUTLASS.

```{warning}
The utilization shown in the table below might be different than what you see online,
but the differences can likely be explained by a different input data distribution.
All our benchmarks here use arrays with iid normal float16 entries, which turn out
to be one of the slower distributions you can choose. You can reproduce
the numbers for yourself by running [our test file](https://github.com/jax-ml/jax/blob/main/tests/pallas/mgpu_examples_test.py) after changing the `BENCHMARK` variable to `True`.

**tl;dr** don't believe matmul benchmarks if they don't specify input data distribution.
```

| Implementation                  | TensorCore utilization | % of cuBLAS utilization |
|---------------------------------|------------------------|-------------------------|
| 0. Basic kernel                 | 37.62%                 | 59.4%                   |
| 1. Warp specialization          | 45.47%                 | 71.7%                   |
| 2. Tiled epilogue               | 55.82%                 | 88.1%                   |
| 3. Collective (2CTA) MMA        | 59.41%                 | 93.7%                   |
| 4. Persistent kernel            | 61.46%                 | 97.0%                   |
| 5. Dedicated epilogue warpgroup | 63.38%                 | 100.0%                  |
| 6. Grid tiling                  | 69.44%                 | 109.6%                  |
| cuBLAS                          | 63.38%                 | 100.0%                  |
| CUTLASS                         | 69.30%                 | 109.3%                  |

The cuBLAS baseline is obtained by measuring the performace of `jax.dot`. The
CUTLASS performance is measured by taking the best result from the following
`cutlass_profiler` invocation (excluding sparse matmuls):
```
cutlass_profiler --dist=gaussian,mean:0,stddev:1,scale:-1 --output=results.csv --accumulator-type=f32 --m=4096 --k=4096 --n=8192 --kernels='*sm100*' --A=f16 --B=f16 --C=void --D=f16
```

At each step, we will showcase either the full implementation of the kernel, or
the difference between the code listings shown in the previous and current steps.
Full implementations can be found in [our test file](https://github.com/jax-ml/jax/blob/main/tests/pallas/mgpu_examples_test.py). You can also find the a full standalone
optimized kernel implementation [in the Pallas ops package](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py).

## 0. Basic kernel

We begin with a simple single-CTA (block) single-warpgroup example.
For convenience, we split the tuning parameters of the kernel into a separate
class:

```python
@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
```

`tile_m`, `tile_n` and `tile_k` specify the size of the matmul performed at
every step of the pipeline. In general, `tile_k` should ideally be equal to
128 divided by the bytewidth of the input element type. `max_concurrent_steps`
specifies the depth of memory prefetch in the compute/memory pipeline, which is
frequently called the number of stages in other implementations.

The kernel implementation begins with a bit of setup:

```python
def matmul0(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps
```

We unpack the config variables for easier access, set the tiling and swizzling
transforms to get the SMEM data format to match [what's expected by MMA instructions](#memory-space-a-b-mma).

The kernel implementation itself is relatively short. The first part sets up
a [compute/memory pipeline](./pipelining.md) using {py:func}`plgpu.emit_pipeline <jax.experimental.pallas.mosaic_gpu.emit_pipeline>`. At each step, the compute function (`do_mma`) consumes a
`(tile_m, tile_k)` slice of LHS and `(tile_k, tile_n)` slice of RHS. As mentioned
before, we specify `transforms`, as well `delay_release=1`. This last parameter
ensures that the input windows (`a_smem`, `b_smem`) passed into `do_mma` will not
be overwritten at least until the next invocation of `do_mma` completes. This is
necessary because we only await the completion of the MMA from the one step
in the following step, which is why `arrive_barrier_slot` and `wait_barrier_slot`
flip between 0 and 1 at each invocation.

```python
  def kernel(a_gmem, b_gmem, out_gmem, acc_tmem, acc_smem, consumed_barriers):
    mi = lax.axis_index("m")
    ni = lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    def do_mma(idxs, a_smem, b_smem):
      (ki,) = idxs
      arrive_barrier_slot = ki % 2
      wait_barrier_slot = 1 - arrive_barrier_slot
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier=consumed_barriers.at[arrive_barrier_slot],
          accumulate=(ki > 0),
      )
      plgpu.barrier_wait(consumed_barriers.at[wait_barrier_slot])

    # Make sure the wait succeeds in the first iteration.
    plgpu.barrier_arrive(consumed_barriers.at[1])
    block_kwargs = dict(transforms=transforms, delay_release=1)
    plgpu.emit_pipeline(
      do_mma,
      in_specs=[
          plgpu.BlockSpec((tile_m, tile_k), lambda ki: (mi, ki), **block_kwargs),
          plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, ni), **block_kwargs),
      ],
      grid=(k_iters,),
      max_concurrent_steps=max_concurrent_steps,
    )(a_gmem, b_gmem)
```

The kernel itself ends with an epilogue. We await the completion of the last MMA
issued by the pipeline before doing anything. Then, we load the final accumulator
from TMEM, write it to SMEM ([remembering to call `plgpu.commit_smem`](#commit-smem)),
and copy it back to GMEM using TMA.

```python
  def kernel(...):
    ...  # compute pipeline as above
    final_barrier = 1 - (k_iters % 2)
    plgpu.barrier_wait(consumed_barriers.at[final_barrier])
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

What remains is to actually turn the kernel into a function that can be called
with JAX arrays. We use {py:func}`plgpu.kernel <jax.experimental.pallas.mosaic_gpu.kernel>`
for that. The grid is for now simply 2D and iterates over the output tiles. We
allocate intermediates used by the kernel:
1. The TMEM buffer used as an accumulator
2. The SMEM buffer used to stage the accumulator before its copy to GMEM
3. The barrier used to await the completion of MMA operations.

```python
def matmul0(a, b, config):
  ... # Setup code from the first snippet
  def kernel(...):
    ... # The whole kernel body

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_shapes=dict(
        acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
        acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        consumed_barriers=plgpu.Barrier(
          num_arrivals=1, num_barriers=2, orders_tensor_core=True
        ),
      )
  )
  return f(a, b)
```

Omitting the setup code, that's just 50 lines! Unfortunately, it's not very
fast just yet, but it does achieve half the utilization of cuBLAS already!

## 1. Warp specialization

```{note}
Recall that on Blackwell a single Pallas:MGPU thread of execution corresponds to
a warpgroup of CUDA lanes/threads.
```

The kernel above uses a single warpgroup to do everything: from fetching the data,
through issuing MMA operations, to storing the results into GMEM. While one would
think that the asynchronicity in TensorCore execution should allow us to overlap
the overheads of async copies (TMA) and control-flow, it does not seem to be the
case.

A common solution to this problem in the Hopper generation of GPUs was to utilize
_warpgroup_ specialization. In Pallas terms, `plgpu.kernel` can be called with
`num_threads=2`, meaning that each program in the grid would result in two calls
to the body. The thread index is then often queried using `lax.axis_index` and
used to select one of multiple different roles, such as _only_ issuing async
copies or _only_ running the MMA operations.

This solution also works in the Blackwell generation, but it is in fact even
simpler. Since both the async copy (TMA) as well as the `tcgen05` MMA instruction [only require a single CUDA lane to issue them](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-issue-granularity),
we don't even need to use multiple _warpgroups_. We can simply break up a single
warpgroup into _four warps_ and specialize those!

In Pallas, this can be achieved using `pl.core_map` with a `plgpu.WarpMesh`.
For each Pallas thread that calls such a `core_map`, the body will be invoked
exactly four times. The `core_map` synchronizes all warps both at entry at exit.
Note that only scalar operations are allowed in the body.

This will be the biggest rewrite to this kernel we'll perform in this whole
sequence, which is why we'll list the entire kernel source once again.

```python
def matmul1(a, b, config: TuningConfig):
  ... # Setup code remains unmodified

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])
          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot]
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot]
          )

        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

The kernel has exactly the same structure as before: we first perform the compute,
which is followed by the epilogue. The epilogue remains the same (we only use a
different barrier to await the completion), so we will not discuss it further.

The `plgpu.emit_pipeline` call and the `do_mma` function has been replaced by
a single `pl.core_map` invocation. You can see that immediately after entering
its body, each Pallas thread (now representing a warp!) finds out which of the
four threads it is. We then use thread with index 0 to _only_ issue async
copies that fetch the MMA operands in a loop, while thread with index 1 enters
another loop in which it repeatedly calls `plgpu.tcgen05_mma`.

One interesting aspect here is the synchronization. We keep an array of
`load_barriers`, each tracking progress of an outstanding GMEM->SMEM copy.
The compute thread must await their completion before feeding the respective
operands to the MMA operation. Going in the other direction, the thread responsible
for async copies must await the completion of MMAs that consume operands before
it can overwrite the memory by issuing another async copy. This is tracked through
`consumed_barriers`. Finally, when the compute thread is done issuing all MMA
operations, it calls `plgpu.tcgen05_commit_arrive(mma_done_barrier)`, requesting
the TensorCore to complete the `mma_done_barrier` once all the MMA operations complete.

We can now turn our attention to the `plgpu.kernel` definition. The only difference
to the prior version is that we explicitly allocate two additional SMEM buffers
that hold the MMA operands (previously they were implicitly allocated by
`plgpu.emit_pipeline`), as well as the additional barriers. Note that the
`load_barriers` have `num_arrivals=2`, because we issue two async copies on the
same barrier. `orders_tensor_core` is necessary to specify on barriers that are
meant to indicate the completion of TensorCore operations.

```python
def matmul1(a, b, config: TuningConfig):
  ... # Setup code remains unmodified

  def kernel(...):
    ... # Kernel code above

  f = plgpu.kernel(
      kernel,
      ...,  # Other parameters remain unchanged
      scratch_shapes=dict(
        a_smem=plgpu.SMEM(
            (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
        ),
        b_smem=plgpu.SMEM(
            (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
        ),
        acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
        acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        load_barriers=plgpu.Barrier(
            num_arrivals=2, num_barriers=max_concurrent_steps
        ),
        consumed_barriers=plgpu.Barrier(
            num_arrivals=1,
            num_barriers=max_concurrent_steps,
            orders_tensor_core=True,
        ),
        mma_done_barrier=plgpu.Barrier(
            num_arrivals=1, num_barriers=1, orders_tensor_core=True
        ),
      )
  )
  return f(a, b)
```

This relatively simple modification already gives us a meaningful bump in performance,
getting us up to almost 70% of cuBLAS performance.

## 2. Tiled epilogue

This time, we turn our attention away from the compute portion of the kernel and
instead focus on its epilogue. We can improve its efficiency by pipelining
the copy from TMEM to SMEM together with a transfer from SMEM to GMEM. To do this,
we change our `scratch_shapes` to allocate two smaller buffers instead of an
SMEM window that can hold the entire output (which also decreases our SMEM usage):

```python
def matmul2(a, b, config):
  ... # Setup and kernel code
  f = plgpu.kernel(
      ...
      scratch_shapes=dict(
        ...
        # Previously: plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        acc_smem=plgpu.SMEM(
            (2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms
        ),
        ...
      )
  )
```

Then, in the kernel, we loop over the output columns in chunks of `epilogue_tile_n`,
and progressively send out the output to GMEM:

```python
def matmul2(a, b, config):
  ... # Setup code remains unchanged

  def kernel(...):
    ... # Compute part remains unchanged

    plgpu.barrier_wait(mma_done_barrier)
    out_gmem_window = out_gmem.at[m_slice, n_slice]
    for ni in range(tile_n // config.epilogue_tile_n):
      acc_smem_ni = acc_smem.at[ni % 2]
      ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
      # Make sure that previous copy is done before we overwrite.
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

## 3. Collective (2CTA) MMA

If you benchmark our latest kernel, you'll quickly find out that it can't utilize
its compute units well, because they are constantly waiting on the memory to deliver
the MMA operands. This means that our kernel is memory bound, because it has too
low _arithmetic intensity_: the number of flops we perform for each byte we load
is too small.

One very effective trick of the Blackwell architecture that allows us to double
our arithmetic intensity are [collective MMAs](#collective-mma).
The core idea is quite simple: we use a cluster of two blocks (on two SMs) to
compute a single matmul. Each block only loads half of each operand, but the MMA
operation exchanges the data from SMEM of each block as its running.

We'll start with the kernel configuration changes again:

```python
def matmul3(a, b, config):
  ...  # Setup code
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  ... # Setup code and kernel

  f = plgpu.kernel(
      ...
      grid=(m_iters, n_iters),
      ...
      cluster=(2,),
      cluster_names=("cluster",),
      scratch_shapes=dict(
          ...
          # Previously: plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_tmem=plgpu.TMEM(
              (tile_m, cluster_tile_n), jnp.float32, collective=True
          ),
          ...
      )
  )
```

We add the `cluster` parameter to `plgpu.kernel` to indicate that we intend to
have pairs of programs collaborate (as CUDA block clusters). We also append
`collective=True` to our TMEM allocation, to ensure that it will be allowed to
be used by collective MMAs and double its number of columns (to `cluster_tile_n`).

Another notable change is that our pair of blocks will ultimately compute a
4x larger output tile, which is why we shrink the grid correspondingly.

We first update the entry of the kernel:

```python
  def kernel(...):
    is_lead_block = lax.axis_index("cluster") == 0
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
    n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
```

The only changes here are that we use `cluster_tile_m` and `cluster_tile_n` to
compute the slice of the output the two blocks will collectively compute, and
we also check if the current invocation corresponds to the first (leader) block
in the cluster. This is important, because _only the leader block is supposed to
issue MMA instructions_:

```python
    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          ...  # Wait for the data to be consumed, as previously.
          plgpu.copy_gmem_to_smem(
              ..., collective_axes="cluster", partitioned_axis=0
          )
          plgpu.copy_gmem_to_smem(
              ..., collective_axes="cluster", partitioned_axis=1
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
      def _compute():
        def _loop_body(ki, _):
          ...  # Wait for the data to arrive, as previously.
          plgpu.tcgen05_mma(
              ...,
              collective_axis="cluster",
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier, collective_axis="cluster")
```

You can see a few modifications here. First of all, both blocks must issue the
async copies. In both blocks we request a copy of the full window for the whole
cluster, but the addition of `collective_axes="cluster"` indicates that the load
is performed jointly by both blocks. `partitioned_axis=` specifies which axis of
the operand should be split across the cluster. We split the LHS rows and RHS
columns.

```{warning}
A partitioned collective copy only completes the barrier passed in to `copy_gmem_to_smem`
in the leader block of the cluster! This is why you will see the kernel never
awaits the loads in the second block.
```

Secondly, as mentioned before, we additionally predicate the `_compute` body so
that only the leader block runs MMA instructions. All `tcgen05` calls additionally
get a `collective_axis=` argument, to indicate that the completion of MMAs should
complete the barriers in both blocks in the cluster.

Finally, we apply a small modification to the epilogue. Even though the two
blocks in the cluster collectively compute a result of shape `(cluster_tile_m, cluster_tile_n)`,
each individual block only holds a result of shape `(tile_m, cluster_tile_n)`.
We change the output slicing code to need to slice out the right `out_gmem_window`:

```python
def matmul3(a, b, config):
  ...
  def kernel(...):
    ... # Compute

    plgpu.barrier_wait(mma_done_barrier)
    out_m_index = m_index * 2 + lax.axis_index("cluster")
    out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
    out_gmem_window = out_gmem.at[out_m_slice, n_slice]
    for ni in range(cluster_tile_n // config.epilogue_tile_n):
      ...

  ...
```

## 4. Persistent kernel

Our next step is to make the kernel persistent. This means that we'll only
launch however many clusters we can actually run concurrently on the GPU (SM
count divided by 2), and we'll have each cluster loop over a fixed number of
output tiles.  This technique allows us to better amortize block
(de)initialization costs (since they are only performed once on each SM) and
achieve a small degree of overlap between the SMEM to GMEM copy in the epilogue
with the compute on the next output tile.

```python
def matmul4(a, b, config):
  ...

  num_sms = jax.extend.backend.get_default_device().core_count
  f = plgpu.kernel(
      ...
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      ...
  )
```

The change is relatively simple. We utilize the {py:func}`plgpu.nd_loop <jax.experimental.pallas.mosaic_gpu.nd_loop>`
helper to specify that our iteration space is `(m_iters, n_iters)`, but we also
request that it should be split accross the cluster grid using the `collective_axes=`
argument.

```python
def matmul4(a, b, config):
  ...

  def kernel(...):
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters, n_iters), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      m_index, n_index = loop_info.index
      m_slice = ...
      n_slice = ...

      ...  # Compute + epilogue
```

The only meaningful modification in the compute portion of the kernel body is
to ensure that the first few waits on `consumed_barriers` in the memory warp
are only skipped when processing the first output tile (as indicated by
`loop_info.local_index == 0`). When processing the second (or later) tile, the SMEM buffers
were used to compute the previous output tile, so we need to ensure that those
computations have completed before we overwrite them:

```python
def matmul4(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ...

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == 0)
        def _memory():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
            def _():  # Make sure the data has been consumed before overwriting.
              plgpu.barrier_wait(consumed_barriers.at[slot])
```

Finally, we modify the kernel epilogue by appending a single line:
```python
def matmul4(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ...  # Compute + epilogue
      plgpu.wait_load_tmem()  # Load must complete before MMA can overwrite TMEM.
```

As the comment indicates, since [TMEM loads are asynchronous](#tmem-loads),
we must await their completion before we move on to the next output tile and
overwrite our TMEM allocation by issuing another MMA.

## 5. Dedicated epilogue warpgroup

While persistence was useful by itself, it also unlocks another optimization.
When the single Pallas thread in our kernel finishes the compute portion of the
kernel, it performs the entire epilogue. However, this means that it can't issue
any more work for the TensorCore until it's done!

This leads us to a simple solution: just use 2 Pallas threads (warpgroups)! The
first one will only focus on fetching the MMA operands and issuing the MMA
operations, while the second one will only perform the epilogue! Of course, to
enable them to run concurrently, we need to double-buffer the TMEM used for
the accumulator, and use additional barriers to synchronize:

```python
def matmul5(a, b, config):
  ...

  f = plgpu.kernel(
      ...,
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          ...
          # Previously: plgpu.TMEM((tile_m, cluster_tile_n), jnp.float32, collective=True),
          acc_tmem=plgpu.TMEM(
              (tile_m, 2 * cluster_tile_n), jnp.float32, collective=True
          ),
          ...
          # mma_done_barrier (now 2 barriers) + a new store_done_barrier (also 2 barriers)
          # Previously: plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      ),
  )
```

The kernel begins similarly to what we had before. We renamed `acc_tmem` to `acc_tmem_slots`
and switch between its halves as we step through the loop over the output tiles:

```python
def matmul(a, b, config):
  ...

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem_slots, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = ...

    @plgpu.nd_loop(...)
    def _mn_loop(...):
      ...
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      acc_tmem = acc_tmem_slots.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      ...
```

The compute portion is additionally predicated on `wg_idx == 0`. There are also
two important changes to how we use the barriers. First of all, if we want to
reuse our TMEM allocation for MMA (which happens only for `loop_info.local_index >= 2`),
we need to wait on the `store_done_barrier` for the TMEM half we want to reuse
(as indicated by `acc_slot`). Secondly, once we want to request the TensorCore
to arrive on the `mma_done_barrier` upon completion, we again need to select one
of the two barriers that corresponds to the currently used half of TMEM.

```{warning}
Note that even though only one of the blocks in the cluster issues MMAs, they
both await the `store_done_barrier`. This is only necessary, because arriving on
the same barrier twice without a `wait` in between sometimes leads to hardware
assertions.
```

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      acc_slot = ...
      acc_tmem = ...

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            ... # Memory code remains unchanged

          # Wait for store to complete (except for the first two steps).
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            ... # Compute loop remains unchanged
            plgpu.tcgen05_commit_arrive(mma_done_barrier.at[acc_slot], collective_axis="cluster")
```

Finally, we modify the epilogue, by only having the second warpgroup execute
it, and by making the warpgroup signal the completion of the store by arriving
on the `store_done_barrier` associated with the half of TMEM it used.

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ... # Compute

      @pl.when(wg_idx == 1)
      def _store_wg():
        ... # Unmodified epilogue
        plgpu.wait_load_tmem()  # Load must complete before we signal.
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
```

## 6. Grid tiling

Our final change to this kernel is to change the order in which we produce the
output blocks to better utilize L2. As mentioned before, the compute units are
extremely fast compared to the memory system and so we could use all the help
we can get to try to keep them busy.

```{note}
This is trick goes by many different names. CUTLASS calls it "rasterization order",
ThunderKittens calls it "supergrouping", while the Triton tutorials call it
"program re-ordering". We use the name "grid tiling".
```

Our strategy for this is inspired by CUTLASS and works as follows. First, you
select which of the two dimensions in your iteration space is the faster changing
(we call it `grid_minor_dim`). Then, you select the tile size along that dimension
(`grid_tile_width`). Instead of traversing the whole minor dimension of the grid
before incrementing the more major index, we do it every time we traverse
`grid_tile_width` elements. Once we run out of elements, we move on to the next
tile. But there's a twist! Instead of jumping to the beginning of the second tile,
we start from the end and work our way back. This ensures that as we switch the
tiles, we can reuse some of the recent blocks of one of the operands.

Since this strategy is so common, we provide a helper for it: {py:func}`plgpu.planar_snake <jax.experimental.pallas.mosaic_gpu.planar_snake>`.
When using the helper, the changes to the kernel are quite trivial:

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    # We now only iterate over a 1D loop (but we still split it across clusters).
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),  # The 2D iteration space.
          config.grid_minor_dim,  # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,  # The width of tiles along the fastest changing dim.
      )
      ... # Rest of the code remains unmodified
```

This simple trick is _incredibly effectful_ and is crucial in achieving state of
the art performance.

## Final kernel

You've reached the end of this tutorial, congratulations! In the previous
sections, we focused only on the differences between the different kernels and
rarely listed the complete source. This is useful to hide the irrelevant details
when extending the implementation, but it can also be helpful to see the full
source. So here it is! The whole implementation is less than 150 lines and
reaches SOTA performance (at least on the shape used in our benchmarks).

```python
def matmul6(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
      n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      mn_acc_tmem = acc_tmem.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
              def _():  # Make sure the data has been consumed before overwriting.
                plgpu.barrier_wait(consumed_barriers.at[slot])
              k_slice = pl.ds(ki * tile_k, tile_k)
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=0
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=1
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          # Wait for store to complete (except for the first two steps).
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
              plgpu.tcgen05_mma(
                  mn_acc_tmem,
                  a_smem.at[slot],
                  b_smem.at[slot],
                  consumed_barriers.at[slot],
                  accumulate=(ki > 0),
                  collective_axis="cluster",
              )
            lax.fori_loop(0, k_iters, _loop_body, None)
            plgpu.tcgen05_commit_arrive(
                mma_done_barrier.at[acc_slot],
                collective_axis="cluster",
            )

      @pl.when(wg_idx == 1)
      def _store_wg():
        # Ensure that copies from the previous mn step have completed.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        out_m_index = m_index * 2 + lax.axis_index("cluster")
        out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
        out_gmem_window = out_gmem.at[out_m_slice, n_slice]
        for ni in range(cluster_tile_n // config.epilogue_tile_n):
          acc_smem_ni = acc_smem.at[ni % 2]
          ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
          # Make sure that previous copy is done before we overwrite.
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          acc_smem_ni[...] = plgpu.async_load_tmem(mn_acc_tmem.at[:, ni_slice]).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_load_tmem()  # Load must complete before we signal.
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      cluster=(2,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
          ),
          acc_tmem=plgpu.TMEM(
              (tile_m, 2 * cluster_tile_n), jnp.float32, collective=True
          ),
          acc_smem=plgpu.SMEM(
              (2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms
          ),
          load_barriers=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      )
  )
  return f(a, b)
```
