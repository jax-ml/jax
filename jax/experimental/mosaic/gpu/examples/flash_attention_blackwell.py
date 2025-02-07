import os
import contextlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import ShapeDtypeStruct as SDS
from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds, utils, profiler, fori, TileTransform, TransposeTransform, create_descriptor
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import math
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector

def create_smem_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | None,
    memory_space: int | None = None,
):
  i64 = ir.IntegerType.get_signless(64)

  hopper_desc = create_descriptor(
    memref_arg, leading_byte_offset, stride_byte_offset, swizzle, memory_space=3)
  blackwell_bit = arith.shli(c(1,i64), c(46,i64))
  blackwell_desc = llvm.or_(hopper_desc, blackwell_bit)
  return blackwell_desc


def smem_descriptor_increment_address(desc, nbytes):
  i64 = ir.IntegerType.get_signless(64)
  return arith.addi(desc, arith.shrui(c(nbytes,i64), c(4,i64)))


def create_blackwell_fp16_mma_descriptor(
    m: int,
    n: int,
    dtype,
    transpose_b: bool = False,
):
  i32 = ir.IntegerType.get_signless(32)
  desc = c(0,i32)
  def fieldset(val, bit):
    field = arith.shli(c(val,i32), c(bit, i32))
    nonlocal desc
    desc = arith.ori(desc, field)
  # encoding is dependent on .kind::<foo>, below is for .kind::f16
  #  1: 0 - sparsity selector if sparsity is enabled
  #  2    - sparsity: dense = 0, sparse = 1
  #  3    - saturate for integer types, irrelevant for fp
  #  5: 4 - output dtype
  fieldset(1, 4)
  #  6    - reserved
  #  9: 7 - A dtype: f16 = 0, b16 = 1
  if dtype == jnp.bfloat16:
    fieldset(1, 7)
  # 12:10 - B dtype: f16 = 0, b16 = 1
  if dtype == jnp.bfloat16:
    fieldset(1, 10)
  # 13    - negate A
  # 14    - negate B
  # 15    - transpose A
  # 16    - transpose B
  if transpose_b:
    fieldset(1, 16)
  # 22:17 - N >> 3
  fieldset(n >> 3, 17)  # TODO validate field width
  # 23    - reserved
  # 28:24 - M >> 4
  fieldset(m >> 4, 24)
  # 29    - reserved
  # 31:30 - max shift under .ws: irrelevant here
  return desc

WARPSIZE = 32
WARPGROUPSIZE = 4 * WARPSIZE
blackwell_mma_fp16_k = 16

def bytecount(shape, dtype):
  return int(np.prod(shape) * dtype.dtype.itemsize)

def attention(q, k, v):
  assert len(q.shape) == 3
  assert len(k.shape) == 3
  assert len(v.shape) == 3

  assert q.dtype == k.dtype
  assert k.dtype == v.dtype
  assert q.dtype in (jnp.float16, jnp.bfloat16)
  dtype = q.dtype

  assert k.shape == v.shape

  assert q.shape[-1] == k.shape[-1]  # head_dim

  num_q_heads, q_seq_len, head_dim = q.shape
  num_kv_heads, kv_seq_len, _ = k.shape

  # MHA
  assert num_q_heads == num_kv_heads

  q_seq_len_tile = 128
  kv_seq_len_tile = 128

  assert q_seq_len % q_seq_len_tile == 0
  assert kv_seq_len % kv_seq_len_tile == 0

  f = build_kernel(
      q_seq_len=q_seq_len,
      q_seq_len_tile=q_seq_len_tile,
      kv_seq_len=kv_seq_len,
      kv_seq_len_tile=kv_seq_len_tile,
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      dtype=dtype,
  )
  return f(q, k, v)


def build_kernel(q_seq_len, q_seq_len_tile, kv_seq_len, kv_seq_len_tile, num_q_heads, num_kv_heads, head_dim, dtype):
  swizzle = 128  # 128 bytes
  def kernel(ctx, q_gmem, k_gmem, v_gmem, o_gmem, smem):
    i1 = ir.IntegerType.get_signless(1)
    i32 = ir.IntegerType.get_signless(32)
    eltype = q_gmem.type.element_type
    assert eltype in (ir.F16Type.get(), ir.BF16Type.get())
    f32 = ir.F32Type.get()
    index = ir.IndexType.get()
    ptr6 = ir.Type.parse("!llvm.ptr<6>")
    tidx = gpu.thread_id(gpu.Dimension.x)
    warpid = arith.shrui(tidx, c(5,index))
    warpgroup_warp0 = arith.muli(c(4,index), arith.divui(warpid, c(4,index)))
    wg_tidx = arith.subi(tidx, arith.muli(warpgroup_warp0, c(WARPSIZE,index)))
    tma_warp = 0
    q_smem, k_smem, v_smem, (q_full_barrier, k_full_barrier, v_full_barrier, qk_done_barrier, pv_done_barrier), tmem_addr, tmem_addr_p = smem
    consumer_warp_range = (1*WARPGROUPSIZE//WARPSIZE, 2*WARPGROUPSIZE//WARPSIZE)
    q_seq_len_tile_i = gpu.block_id(gpu.Dimension.x)
    q_head_i = gpu.block_id(gpu.Dimension.y)
    q_seq_len_tile_offset = arith.muli(q_seq_len_tile_i, c(q_seq_len_tile,index))
    k_steps = kv_seq_len//kv_seq_len_tile  # number of outer loop iterations

    @contextlib.contextmanager
    def only_warp(i):
      is_warp_i = arith.cmpi(arith.CmpIPredicate.eq, warpid, c(i,index))
      with ir.InsertionPoint(scf.IfOp(is_warp_i).then_block):
        yield
        scf.yield_([])

    @contextlib.contextmanager
    def only_warp_range(a, b):
      gea = arith.cmpi(arith.CmpIPredicate.uge, warpid, c(a,index))
      ltb = arith.cmpi(arith.CmpIPredicate.ult, warpid, c(b,index))
      predicate = arith.andi(gea, ltb)
      with ir.InsertionPoint(scf.IfOp(predicate).then_block):
        yield
        scf.yield_([])

    @contextlib.contextmanager
    def single_warp_thread():
      elected = nvvm.elect_sync(i1)
      with ir.InsertionPoint(scf.IfOp(elected).then_block):
        yield
        scf.yield_([])

    with only_warp(tma_warp):
      with single_warp_thread():
        # initiate Q load
        ctx.async_copy(
          src_ref=q_gmem,
          dst_ref=q_smem,
          gmem_slice=(q_head_i, ds(q_seq_len_tile_offset, q_seq_len_tile), ds(0,head_dim)),
          gmem_transform=TileTransform(q_tiling),
          swizzle=swizzle,
          barrier=q_full_barrier,
          arrive=True,
          uniform=False,
        )
        @fori(c(k_steps,index), None)
        def kv_load_loop_body(ki, carrys):
          should_wait = arith.cmpi(arith.CmpIPredicate.ne, ki, c(0,index))
          with ir.InsertionPoint(scf.IfOp(should_wait).then_block):
            qk_done_barrier.wait()
            scf.yield_([])
          kv_seq_len_tile_offset = arith.muli(ki, c(kv_seq_len_tile,index))
          ctx.async_copy(
            src_ref=k_gmem,
            dst_ref=k_smem,
            gmem_slice=(q_head_i, ds(kv_seq_len_tile_offset, kv_seq_len_tile), ds(0,head_dim)),
            gmem_transform=TileTransform(kv_tiling),
            swizzle=swizzle,
            barrier=k_full_barrier,
            arrive=True,
            uniform=False,
          )
          with ir.InsertionPoint(scf.IfOp(should_wait).then_block):
            pv_done_barrier.wait()
            scf.yield_([])
          ctx.async_copy(
            src_ref=v_gmem,
            dst_ref=v_smem,
            gmem_slice=(q_head_i, ds(kv_seq_len_tile_offset, kv_seq_len_tile), ds(0,head_dim)),
            gmem_transform=(TileTransform(kv_tiling), TransposeTransform((1, 0, 2, 3))),
            swizzle=swizzle,
            barrier=v_full_barrier,
            arrive=True,
            uniform=False,
          )

    with only_warp_range(*consumer_warp_range):
      with ctx.named_region("TMEM alloc + Q TMA wait"):
        tmem_addr_addr = utils.memref_ptr(tmem_addr, memory_space=3)
        tmem_addr_p_addr = utils.memref_ptr(tmem_addr_p, memory_space=3)
        ncols_i = max(kv_seq_len_tile, head_dim, 32)
        ncols = c(ncols_i, i32)
        ncols_p_i = max(32,kv_seq_len_tile)
        ncols_p = c(ncols_p_i,i32)
        with only_warp(consumer_warp_range[0]):
          nvvm.tcgen05_alloc(tmem_addr_addr, ncols)
          nvvm.tcgen05_alloc(tmem_addr_p_addr, ncols_p)
          nvvm.tcgen05_relinquish_alloc_permit()
        o = vector.splat(ir.VectorType.get([head_dim], f32), c(0,f32))
        l = arith.constant(f32, 0.0)
        m = arith.constant(f32, float("-inf"))
        q_full_barrier.wait()
      @fori(c(k_steps,index), (o, l, m))
      def kv_loop_body(ki, carrys):
        o, l, m = carrys
        with ctx.named_region("QK"):
          with only_warp(consumer_warp_range[0]):
            with single_warp_thread():
              tmem_addr_value = llvm.load(ptr6, tmem_addr_addr)
              idesc = create_blackwell_fp16_mma_descriptor(q_seq_len_tile, kv_seq_len_tile, dtype)
              qdesc = create_smem_descriptor(
                q_smem, leading_byte_offset=16, stride_byte_offset=1024, swizzle=128)
              kdesc = create_smem_descriptor(
                k_smem, leading_byte_offset=16, stride_byte_offset=1024, swizzle=128)
              k_full_barrier.wait()
              accumulate = 0
              blackwell_mma_fp16_k = 16
              for i in range(head_dim//blackwell_mma_fp16_k):
                nvvm.tcgen05_mma("f16", "cta_1", tmem_addr_value, qdesc, kdesc, idesc, enable_input_d=c(accumulate,i1))
                accumulate = 1
                increment = (q_seq_len_tile*4 - 3)*blackwell_mma_fp16_k*2 if i%4==3 else blackwell_mma_fp16_k*2
                qdesc = smem_descriptor_increment_address(qdesc, increment)
                increment = (kv_seq_len_tile*4 - 3)*blackwell_mma_fp16_k*2 if i%4==3 else blackwell_mma_fp16_k*2
                kdesc = smem_descriptor_increment_address(kdesc, increment)
              nvvm.tcgen05_commit_arrive(qk_done_barrier.get_ptr())
          qk_done_barrier.wait()
        with ctx.named_region("Softmax"):
          warpid_within_warpgroup = arith.subi(warpid, warpgroup_warp0)
          warp_within_warpgroup_tid0 = arith.muli(warpid_within_warpgroup, c(WARPSIZE,index))
          tmem_lane = arith.shli(warp_within_warpgroup_tid0, c(16,index))
          tmem_addr_value = memref.load(tmem_addr, [c(0,index)])
          tmem_addr32_warp = arith.ori(tmem_addr_value, arith.index_cast(i32,tmem_lane))
          tmem_ptr = llvm.inttoptr(ptr6, tmem_addr32_warp)
          vector_i32 = nvvm.tcgen05_ld(ir.VectorType.get([kv_seq_len_tile], i32), shape="shape_32x32b", num=kv_seq_len_tile, tmem_addr=tmem_ptr)
          vector_f32 = vector.bitcast(ir.VectorType.get(vector_i32.type.shape, f32), vector_i32)
          m_old = m
          m = arith.maximumf(m_old, vector.reduction(f32, vector.CombiningKind.MAXIMUMF, vector_f32))
          vector_f32 = arith.subf(vector_f32, vector.splat(vector_f32.type, m))
          p = math.exp(vector_f32)
          l = arith.addf(
            arith.mulf(math.exp(arith.subf(m_old, m)),
                       l),
            vector.reduction(f32, vector.CombiningKind.ADD, p))
          p16 = arith.truncf(ir.VectorType.get(p.type.shape, eltype), p)
          tmem_addr_p_value = memref.load(tmem_addr_p, [c(0,index)])
          tmem_addr32_p_warp = arith.ori(tmem_addr_p_value, arith.index_cast(i32,tmem_lane))
          tmem_ptr_p = llvm.inttoptr(ptr6, tmem_addr32_p_warp)
          p16_packed = vector.bitcast(ir.VectorType.get([p16.type.shape[0]//2], i32), p16)
          nvvm.barrier(barrier_id=c(2,i32), number_of_threads=c(128,i32))
          nvvm.tcgen05_st(shape="shape_32x32b", num=kv_seq_len_tile//2, tmem_addr=tmem_ptr_p, r=p16_packed, unpack=False)
          nvvm.tcgen05_wait(kind="store")
        with ctx.named_region("V TMA wait"):
          v_full_barrier.wait()
        with ctx.named_region("PV"):
          with only_warp(consumer_warp_range[0]), single_warp_thread():
            tmem_addr_value = llvm.load(ptr6, tmem_addr_addr)
            tmem_addr_p_value = llvm.load(ptr6, tmem_addr_p_addr)
            idesc = create_blackwell_fp16_mma_descriptor(q_seq_len_tile, head_dim, dtype, transpose_b=True)
            vdesc = create_smem_descriptor(
              v_smem, leading_byte_offset=kv_seq_len_tile*64*2, stride_byte_offset=1024, swizzle=128)

            accumulate = 0
            blackwell_mma_fp16_k = 16
            for _ in range(kv_seq_len_tile//blackwell_mma_fp16_k):
              nvvm.tcgen05_mma("f16", "cta_1", tmem_addr_value, tmem_addr_p_value, vdesc, idesc, enable_input_d=c(accumulate,i1))
              accumulate = 1
              tmem_addr_p_value = llvm.inttoptr(ptr6, arith.addi(llvm.ptrtoint(i32, tmem_addr_p_value), c(blackwell_mma_fp16_k//2,i32))) # divide by 2 because each 32-bit wide column has 2 fp16 elements packed
              increment = 64*blackwell_mma_fp16_k*2  # 2 bytes per element
              vdesc = smem_descriptor_increment_address(vdesc, increment)
            nvvm.tcgen05_commit_arrive(pv_done_barrier.get_ptr())
          pv_done_barrier.wait()
        with ctx.named_region("O update"):
          nvvm.barrier(barrier_id=c(1,i32), number_of_threads=c(128,i32))
          if head_dim == 256:
            # can only load up to 128 columns from tmem, so we do it twice
            # and merge the partial results into one vector
            num = head_dim//2  # load 128 columns twice

            tmem_ptr_0 = tmem_ptr
            pv_i32_0 = nvvm.tcgen05_ld(ir.VectorType.get([num], i32), shape="shape_32x32b", num=num, tmem_addr=tmem_ptr_0)
            pv_f32_0 = vector.bitcast(ir.VectorType.get(pv_i32_0.type.shape, f32), pv_i32_0)

            tmem_ptr_1 = llvm.inttoptr(tmem_ptr_0.type, arith.addi(llvm.ptrtoint(i32, tmem_ptr_0), c(num,i32)))
            pv_i32_1 = nvvm.tcgen05_ld(ir.VectorType.get([num], i32), shape="shape_32x32b", num=num, tmem_addr=tmem_ptr_1)
            pv_f32_1 = vector.bitcast(ir.VectorType.get(pv_i32_1.type.shape, f32), pv_i32_1)

            pv_f32 = vector.splat(ir.VectorType.get([head_dim], f32), c(0,f32))
            pv_f32 = vector.insert_strided_slice(pv_f32_0, pv_f32, [  0], [1])
            pv_f32 = vector.insert_strided_slice(pv_f32_1, pv_f32, [num], [1])
          else:
            pv_i32 = nvvm.tcgen05_ld(ir.VectorType.get([head_dim], i32), shape="shape_32x32b", num=head_dim, tmem_addr=tmem_ptr)
            pv_f32 = vector.bitcast(ir.VectorType.get(pv_i32.type.shape, f32), pv_i32)
          mdiff = arith.subf(m_old, m)
          correction_term = math.exp(mdiff)
          o = arith.addf(arith.mulf(o,
                                    vector.splat(o.type, correction_term)),
                         pv_f32)
        return o, l, m

      o, l, m = kv_loop_body.results

      with ctx.named_region("O store"):
        o = arith.divf(o, vector.splat(o.type, l))
        o_f16 = arith.truncf(ir.VectorType.get(o.type.shape, eltype), o)
        vector.store(o_f16, o_gmem, [q_head_i, arith.addi(q_seq_len_tile_offset,wg_tidx), c(0,index)])
        tmem_addr_value = llvm.load(ptr6, tmem_addr_addr)
        tmem_addr_p_value = llvm.load(ptr6, tmem_addr_p_addr)

  q_shape = SDS((num_q_heads, q_seq_len, head_dim), dtype)
  k_shape = v_shape = SDS((num_kv_heads, kv_seq_len, head_dim), dtype)
  in_shape = (
    q_shape,
    k_shape,
    v_shape,
  )
  out_shape = q_shape

  q_shape_tile = (q_seq_len_tile, head_dim)
  k_shape_tile = v_shape_tile = (kv_seq_len_tile, head_dim)
  q_tiling = (q_seq_len_tile, 64)  # 64 * 2B = 128B
  kv_tiling = (kv_seq_len_tile, 64)  # 64 * 2B = 128B

  smem_shape = (
    SDS(utils.tile_shape(q_shape_tile,  q_tiling), dtype),
    SDS(utils.tile_shape(k_shape_tile, kv_tiling), dtype),
    SDS((head_dim//64, 1, kv_seq_len_tile, 64), dtype),
    tuple(mgpu.Barrier(arrival_count=1) for _ in range(5)),
    SDS((1,), np.uint32),  # tmem_addr
    SDS((1,), np.uint32),  # tmem_addr_p
  )
  profile = False
  prof_spec = profiler.ProfilerSpec(4096) if profile else None
  if profile:
    os.environ['TEST_UNDECLARED_OUTPUTS_DIR'] = '.'

  grid = (q_seq_len//q_seq_len_tile, num_q_heads, 1)
  block = (2*WARPGROUPSIZE, 1, 1)

  with mlir.make_ir_context(), ir.Location.unknown():
    f = mgpu.as_gpu_kernel(
      body=kernel,
      grid=grid,
      block=block,
      in_shape=in_shape,
      out_shape=out_shape,
      smem_scratch_shape=smem_shape,
      prof_spec=prof_spec,
    )

  return f


@jax.jit
def ref(q, k, v):
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)
  num_q_heads, q_seq_len, head_dim = q.shape
  num_kv_heads, _, _ = k.shape
  q_reshaped = q.reshape(
      num_kv_heads, num_q_heads // num_kv_heads, q_seq_len, head_dim)
  logits = jnp.einsum("xhqc,xkc->xhqk", q_reshaped, k)
  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  return jnp.einsum("xhqk,xkc->xhqc", weights, v).reshape(*q.shape)

# useful for debugging
def repro(q, k, v):
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)

  o = jnp.zeros((q_seq_len_tile, head_dim), dtype=jnp.float32)
  l = jnp.zeros((q_seq_len_tile,        1), dtype=jnp.float32)
  m = jnp.full ((q_seq_len_tile,        1), dtype=jnp.float32, fill_value=-jnp.inf)

  print(f"{q.shape=}")
  print(f"{k.shape=}")
  print(f"{v.shape=}")
  print(f"{o.shape=}")
  print(f"{l.shape=}")
  print(f"{m.shape=}")
  for i in range(kv_seq_len//kv_seq_len_tile):
    ki = k[i*kv_seq_len_tile:(i+1)*kv_seq_len_tile, :]
    print(f"{ki.shape=}")
    s = q @ ki.T
    #print(f"{s=}")
    print(f"{s.shape=}")
    m_old = m
    print(f"{m.shape=}")
    m = jnp.maximum(m_old, s.max(axis=1, keepdims=True))
    #print(f"{m=}")
    #print(f"{m.shape=}")
    p = jnp.exp(s - m)
    #print(f"{p.shape=}")
    mdiff = m_old-m
    #print(f"{mdiff[:32, :]=}")
    correction_term = jnp.exp(mdiff)
    #print(f"{correction_term[:32, :]=}")
    l = correction_term*l + p.sum(axis=1, keepdims=True)
    #print(f"{l.shape=}")
    vi = v[i*kv_seq_len_tile:(i+1)*kv_seq_len_tile, :]
    pv = p@vi
    print(f"{pv[:,[0,128]]=}")
    if i:
      o = o*correction_term + pv
    else:
      o = pv
    #print(f"{o[:32, 0]=}")
    #print(f"{o.shape=}")

  o = o / l
  return o

def test(q_seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, dtype):
  print(f"{q_seq_len=}, {kv_seq_len=}, {num_q_heads=}, {num_kv_heads=}, {head_dim=}, {dtype=}")
  kq, kk, kv = jr.split(jr.key(0), 3)
  q = jr.normal(kq, (num_q_heads, q_seq_len, head_dim), dtype=dtype)
  k = jr.normal(kk, (num_kv_heads, kv_seq_len, head_dim), dtype=dtype)
  v = jr.normal(kv, (num_kv_heads, kv_seq_len, head_dim), dtype=dtype)

  atol = 1e-2 if dtype == jnp.bfloat16 else 1e-3
  rtol = 1e-3
  o = attention(q, k, v)
  o_ref = ref(q, k, v)
  np.testing.assert_allclose(
    o, o_ref, atol=atol, rtol=rtol
  )

if __name__ == "__main__":
  num_heads = 4
  test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim= 64, dtype=jnp.float16)
  test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim=128, dtype=jnp.float16)
  # recently the headdim=256 cases started to fail with:
  #  error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: vector.insert_strided_slice
  #  error: Failed creating the llvm::Module.
  #  error: An error happened while serializing the module.
  #test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim=256, dtype=jnp.float16)
  test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim= 64, dtype=jnp.bfloat16)
  test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim=128, dtype=jnp.bfloat16)
  #test(q_seq_len=4*1024, kv_seq_len=4*1024, num_q_heads=num_heads, num_kv_heads=num_heads, head_dim=256, dtype=jnp.bfloat16)

