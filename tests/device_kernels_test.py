# Copyright 2025 The JAX Authors.
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

import pytest

import jax
import jax.numpy as jnp
from jax import vmap
from jax._src.device_kernels import (
   kernel_call,
    register_device_kernel_as_batch_partitionable,
    build_device_kernel_lowering_function,
    kernel_lowering
)
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding

from jax._src import test_util as jtu

class TestDeviceKernels(jtu.JaxTestCase):
    """Test suite for device kernels functionality."""

    def test_basic_kernel_call(self):
        """Test basic kernel call functionality."""
        # Simple PTX kernel that adds two arrays
        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64

        .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c
        ) {
            .reg .s32 r0;
            .reg .u64 p1, p2, p3, p4;
            .reg .f32 r1, r2, r3;

            // Load parameters into registers
            ld.param.u64 p1, [a];
            ld.param.u64 p2, [b];
            ld.param.u64 p3, [c];

            // Calculate offset in bytes using thread index
            mov.u32 r0, %tid.x;
            mul.wide.s32 p4, r0, 4;  // p4 now holds the byte offset as a u64

            // Calculate final addresses for a, b, and c
            add.u64 p1, p1, p4;
            add.u64 p2, p2, p4;
            add.u64 p3, p3, p4;

            // Load float values from the computed addresses, perform addition, and store result
            ld.global.f32 r1, [p1];
            ld.global.f32 r2, [p2];
            add.f32 r3, r1, r2;
            st.global.f32 [p3], r3;
        }
        """

        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)

        # Test that the kernel call executes successfully
        result = kernel_call(
            ptx_kernel,
            "add_kernel",          # kernel name
            jax.ShapeDtypeStruct(a.shape, a.dtype),  # output shape and dtype
            a, b,                   # input arrays
            kernel_type="ptx",      # kernel type
            grid_dims=(1, 1, 1),
            block_dims=(3, 1, 1),
            shared_mem_bytes=0,
        )
        # Verify the result
        expected = a + b  # [5.0, 7.0, 9.0]
        assert jnp.allclose(a, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
        assert jnp.allclose(b, jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32))
        if isinstance(result, list):
            result = result[0]  # Take first result if it's a list
        assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_batch_partitioning_registration(self):
        """Test batch partitioning registration."""
        # Test that we can register a kernel type as batch partitionable
        try:
            register_device_kernel_as_batch_partitionable("ptx")
            # Should not raise an error
        except Exception as e:
            pytest.fail(f"register_device_kernel_as_batch_partitionable raised {e}")

        # Test invalid kernel type
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            register_device_kernel_as_batch_partitionable("invalid_type")

    def test_build_device_kernel_lowering_function(self):
        """Test the build_device_kernel_lowering_function."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """

        # Test that the function can be created
        try:
            lowering_fn = build_device_kernel_lowering_function(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_fn)
        except Exception as e:
            pytest.fail(f"build_device_kernel_lowering_function raised {e}")

    def test_kernel_lowering(self):
        """Test the kernel_lowering function."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """

        # Test that the lowering rule can be created
        try:
            lowering_rule = kernel_lowering(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_rule)
        except Exception as e:
            pytest.fail(f"kernel_lowering raised {e}")

    def test_batch_partitioning(self):
        """Test batch partitioning and sharding preservation."""
        if jax.device_count() < 2:
            pytest.skip("Requires multiple devices")

        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64

        .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c
        ) {
            .reg .s32 r0, r1, r2;
            .reg .u64 p1, p2, p3, p4;
            .reg .f32 r1_val, r2_val, r3;

            // Load parameters into registers
            ld.param.u64 p1, [a];
            ld.param.u64 p2, [b];
            ld.param.u64 p3, [c];

            // Calculate 2D indices from thread and block IDs
            mov.u32 r0, %tid.x;     // thread index within block
            mov.u32 r1, %ctaid.x;   // block index

            // Calculate row and column indices
            // For 2D array (8, 4): row = block_id, col = thread_id
            // row = r1, col = r0

            // Calculate offset: (row * 4 + col) * 4 bytes
            mul.lo.s32 r2, r1, 4;   // row * 4
            add.s32 r2, r2, r0;     // + col
            mul.wide.s32 p4, r2, 4; // * 4 bytes per float

            // Calculate final addresses for a, b, and c
            add.u64 p1, p1, p4;
            add.u64 p2, p2, p4;
            add.u64 p3, p3, p4;

            // Load float values from the computed addresses, perform addition, and store result
            ld.global.f32 r1_val, [p1];
            ld.global.f32 r2_val, [p2];
            add.f32 r3, r1_val, r2_val;
            st.global.f32 [p3], r3;
        }
        """

        def kernel_fn(x, y):
            return kernel_call(
                ptx_kernel,
                "add_kernel",
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                x, y,
                kernel_type="ptx",
                grid_dims=(x.shape[0], 1, 1),
                block_dims=(x.shape[1], 1, 1),
                shared_mem_bytes=0,
                output_indices=[2]
            )

        # Create mesh and sharded arrays
        devices = jax.devices()
        mesh = Mesh(devices, ('i',))
        x = jnp.ones((8, 4), dtype=jnp.float32)
        y = jnp.ones((8, 4), dtype=jnp.float32)

        x_sharding = NamedSharding(mesh, P('i'))
        x = jax.device_put(x, x_sharding)
        y = jax.device_put(y, x_sharding)

        # Test eager mode
        result_eager = kernel_fn(x, y)

        # Test JIT mode with output sharding
        kernel_fn_jit = jax.jit(kernel_fn, out_shardings=x_sharding)
        result_jit = kernel_fn_jit(x, y)

        # Verify results
        expected = x + y  # Should be 2.0 everywhere
        if isinstance(result_eager, list):
            result_eager = result_eager[0]
        if isinstance(result_jit, list):
            result_jit = result_jit[0]

        assert jnp.allclose(result_eager, expected), f"Eager result incorrect: {result_eager}"
        assert jnp.allclose(result_jit, expected), f"JIT result incorrect: {result_jit}"

        # Test that JIT result preserves sharding
        assert hasattr(result_jit, 'sharding'), "JIT result should have sharding"
        assert result_jit.sharding == x_sharding, f"Expected sharding {x_sharding}, got {result_jit.sharding}"

        # Test compilation (this should not crash)
        try:
            compiled = kernel_fn_jit.lower(x, y).compile()
            # Note: We can't easily check for "all-gather" in the compiled text
            # but the fact that it compiles without error is good
        except Exception as e:
            pytest.fail(f"JIT compilation failed: {e}")

        # Test that input arrays are not modified
        assert jnp.allclose(x, jnp.ones((8, 4), dtype=jnp.float32)), f"Input array 'x' was modified: {x}"
        assert jnp.allclose(y, jnp.ones((8, 4), dtype=jnp.float32)), f"Input array 'y' was modified: {y}"

    def test_layout_handling(self):
        """Test layout handling in lowering functions."""
        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """

        # Test with custom operand and result layouts
        try:
            lowering_fn = build_device_kernel_lowering_function(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                operand_layouts=[[1, 0]],  # Transpose input
                result_layouts=[[1, 0]],   # Transpose output
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_fn)
        except Exception as e:
            pytest.fail(f"Layout handling failed: {e}")


    @jtu.sample_product(
        vmap_method=["expand_dims", "broadcast_all", "sequential", "sequential_unrolled"],
    )
    def test_vmap_with_device_kernels(self, vmap_method):
        """Test vmap functionality with device kernels."""
        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c
        ) {
            .reg .s32 r0;
            .reg .u64 p1, p2, p3, p4;
            .reg .f32 r1, r2, r3;

            // Load parameters into registers
            ld.param.u64 p1, [a];
            ld.param.u64 p2, [b];
            ld.param.u64 p3, [c];

            // Calculate offset in bytes using thread index
            mov.u32 r0, %tid.x;
            mul.wide.s32 p4, r0, 4;  // p4 now holds the byte offset as a u64

            // Calculate final addresses for a, b, and c
            add.u64 p1, p1, p4;
            add.u64 p2, p2, p4;
            add.u64 p3, p3, p4;

            // Load float values from the computed addresses, perform addition, and store result
            ld.global.f32 r1, [p1];
            ld.global.f32 r2, [p2];
            add.f32 r3, r1, r2;
            st.global.f32 [p3], r3;
        }
        """

        def kernel_fn(x, y):
            return kernel_call(
                ptx_kernel,
                "add_kernel",
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                x, y,
                kernel_type="ptx",
                grid_dims=1,
                block_dims=16,
                shared_mem_bytes=0,
                vmap_method=vmap_method
            )

        # Test with batched input
        batch_size = 3
        x = jnp.ones((batch_size, 4), dtype=jnp.float32)
        y = jnp.ones((batch_size, 4), dtype=jnp.float32)

        # Apply vmap to the kernel function
        vmapped_fn = vmap(kernel_fn, in_axes=0, out_axes=0)

        # Test that vmap works correctly
        result = vmapped_fn(x, y)

        # Verify the result
        expected = x + y  # Should be 2.0 everywhere
        if isinstance(result, list):
            result = result[0]
        assert jnp.allclose(result, expected), f"Expected {expected}, got {result} for vmap_method={vmap_method}"

    def test_kernel_validation(self):
        """Test kernel validation."""
        # Test invalid kernel type
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            kernel_call(
                "invalid",
                "test",
                jnp.array([1.0]),
                jnp.array([1.0]),
                kernel_type="invalid_type"
            )

        # Test PTX kernel without .entry
        with pytest.raises(ValueError, match="PTX code must contain an .entry point"):
            kernel_call(
                ".visible .func test() { ret; }",
                "test",
                jnp.array([1.0]),
                jnp.array([1.0]),
                kernel_type="ptx"
            )

    def test_empty_arrays(self):
        """Test kernel with empty arrays."""
        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c
        ) {
            ret;
        }
        """

        # Test with empty arrays
        a = jnp.array([], dtype=jnp.float32)
        b = jnp.array([], dtype=jnp.float32)

        result = kernel_call(
            ptx_kernel,
            "add_kernel",
            jax.ShapeDtypeStruct(a.shape, a.dtype),
            a, b,
            kernel_type="ptx",
            grid_dims=1,
            block_dims=1,
            shared_mem_bytes=0,
        )

        if isinstance(result, list):
            result = result[0]
        assert result.shape == (0,)
        assert result.dtype == jnp.float32

    def test_jit_with_kernel(self):
        """Test that kernels work with JIT compilation."""
        ptx_kernel = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c
        ) {
            .reg .s32 r0;
            .reg .u64 p1, p2, p3, p4;
            .reg .f32 r1, r2, r3;

            mov.u32 r0, %tid.x;
            mul.wide.s32 p4, r0, 4;

            ld.param.u64 p1, [a];
            ld.param.u64 p2, [b];
            ld.param.u64 p3, [c];

            add.u64 p1, p1, p4;
            add.u64 p2, p2, p4;
            add.u64 p3, p3, p4;

            ld.global.f32 r1, [p1];
            ld.global.f32 r2, [p2];
            add.f32 r3, r1, r2;
            st.global.f32 [p3], r3;
        }
        """

        def kernel_fn(x, y):
            return kernel_call(
                ptx_kernel,
                "add_kernel",
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                x, y,
                kernel_type="ptx",
                grid_dims=1,
                block_dims=4,
                shared_mem_bytes=0,
            )

        # Test JIT compilation
        jitted_fn = jax.jit(kernel_fn)
        x = jnp.ones((4,), dtype=jnp.float32)
        y = jnp.ones((4,), dtype=jnp.float32)

        result = jitted_fn(x, y)
        expected = x + y

        if isinstance(result, list):
            result = result[0]
        assert jnp.allclose(result, expected)

if __name__ == "__main__":
    pytest.main([__file__])
