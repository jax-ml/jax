# List of XLA compiler flags

<!--* freshness: { reviewed: '2024-08-18' } *-->

## Introduction
This guide gives a brief overview of XLA and how XLA relates to Jax.
For in-depth details please refer to [XLA documentation](https://openxla.org/xla). Then it lists commonly-used XLA compiler flags designed to optimize performance of Jax programs.

## XLA: The Powerhouse Behind Jax
XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that plays a pivotal role in Jax's performance and flexibility. It enables Jax to generate optimized code for various hardware backends (CPUs, GPUs, TPUs) by transforming and compiling your Python/NumPy-like code into efficient machine instructions.

Jax uses XLA's JIT compilation capabilities to transform your Python functions into optimized XLA computations at runtime.

## Configuring XLA in Jax:
You can influence XLA's behavior in Jax by setting XLA_FLAGS environment variables before running your Python script or colab notebook.

For the colab notebooks:

Provide flags using `os.environ['XLA_FLAGS']`:


```python
import os

# Set multiple flags separated by spaces
os.environ['XLA_FLAGS'] = '--flag1=value1 --flag2=value2'
```

For the python scripts:

Specify `XLA_FLAGS` as a part of cli command:

```bash
XLA_FLAGS='--flag1=value1 --flag2=value2'  python3 source.py
```

**Important Notes:**

* Set `XLA_FLAGS` before importing Jax or other relevant libraries. Changing `XLA_FLAGS` after backend initialization will have no effect and given backend initialization time is not clearly defined it is usually safer to set `XLA_FLAGS` before executing any Jax code.
* Experiment with different flags to optimize performance for your specific use case.


**For further information:**
* Complete and up to date documentation about XLA can be found in the official [XLA documentation](https://openxla.org/xla).

* For backends supported by open-source version of XLA (CPU, GPU), XLA flags are defined with their default values in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc), and a complete list of flags could be found [here](https://github.com/openxla/xla/blob/main/xla/xla.proto).
* TPU compiler flags are not part of [OpenXLA](https://github.com/openxla/xla), but commonly-used options are listed below.

* Please note that this list of flags is not exhaustive and is subject to change. These flags are implementation details, and there is no guarantee that they will remain available or maintain their current behavior.
### Common XLA flags
| Flag | Type | Notes |
| ---- | ---- | ----- |
| `xla_dump_to` | String (filepath) | The folder where pre-optimization HLO files and other artifacts will be placed (see [XLA Tools](https://openxla.org/xla/tools)). |
| `xla_enable_async_collective_permute` | TristateFlag (true/false/auto) | Rewrites all collective-permute operations to their asynchronous variants.  When set to `auto`, XLA can turn on async collective based on other configurations or conditions automatically. |
| `xla_enable_async_all_gather` | TristateFlag (true/false/auto) | If set to true, enables async all gather. If `auto`, enables only for platforms that implement async all-gather. The implementation (such as BC-offload or continuation fusion) is chosen based on other flag values. |
| `xla_disable_hlo_passes` | String (comma-separated list of pass names) | Comma-separated list of HLO passes to be disabled. These names must exactly match the pass name (no whitespace around commas). |

### TPU XLA flags
| Flag | Type | Notes |
| ---- | ---- | ----- |
| `xla_tpu_enable_data_parallel_all_reduce_opt` | Boolean (true/false) | Optimization to increase overlap opportunities for DCN (data center networking) all-reduces used for data parallel sharding. |
| `xla_tpu_data_parallel_opt_different_sized_ops` | Boolean (true/false) | Enables pipelining of data parallel ops across multiple iterations even if their output sizes don't match what can be saved in place in the stacked variables. Can increase memory pressure. |
| `xla_tpu_enable_async_collective_fusion` | Boolean (true/false) | Enables the pass which fuses async collective communications with compute ops (output/loop-fusion or convolution) that are scheduled between their -start and -done instructions. |
| `xla_tpu_enable_async_collective_fusion_fuse_all_gather` | TristateFlag (true/false/auto) | Enables fusing all-gathers within the AsyncCollectiveFusion pass. <br>If set to `auto`, it will be enabled based on the target. |
| `xla_tpu_enable_async_collective_fusion_multiple_steps` | Boolean (true/false) | Enables continuing the same async collective in multiple steps (fusions) in the AsyncCollectiveFusion pass. |
| `xla_tpu_overlap_compute_collective_tc` | Boolean (true/false) | Enables the overlap of compute and communication on a single TensorCore, i.e., one core equivalent of MegaCore fusion. |
| `xla_tpu_spmd_rng_bit_generator_unsafe` | Boolean (true/false) | Whether to run RngBitGenerator HLO in a partitioned way, which is unsafe if deterministic results are expected with different shardings on different parts of the computation. |
| `xla_tpu_megacore_fusion_allow_ags` | Boolean (true/false) | Allows fusing all-gathers with convolutions/all-reduces. |
| `xla_tpu_enable_ag_backward_pipelining` | Boolean (true/false) | Pipelines all-gathers (currently megascale all-gathers) backwards through scan loops. |

### GPU XLA flags
| Flag | Type | Notes |
| ---- | ---- | ----- |
| `xla_gpu_enable_latency_hiding_scheduler` | Boolean (true/false) |This flag enables latency hiding schedulers to overlap asynchronous communication with computation efficiently. The default value is False. |
| `xla_gpu_enable_triton_gemm` | Boolean (true/false) | Use Triton-based matrix multiplication. |
| `xla_gpu_graph_level` | Flag (0-3) | The legacy flag for setting GPU graph level. Use xla_gpu_enable_command_buffer in new use cases. 0 = off; 1 = capture fusions and memcpys; 2 = capture gemms; 3 = capture convolutions. |
| `xla_gpu_all_reduce_combine_threshold_bytes` | Integer (bytes) | These flags tune when to combine multiple small AllGather / ReduceScatter / AllReduce into one big AllGather / ReduceScatter / AllReduce to reduce time spent on cross-device communication. For example, for the AllGather / ReduceScatter thresholds on a Transformer-based workload, consider tuning them high enough so as to combine at least a Transformer Layer’s weight AllGather / ReduceScatter. By default, the combine_threshold_bytes is set to 256. |
| `xla_gpu_all_gather_combine_threshold_bytes` | Integer (bytes) | See xla_gpu_all_reduce_combine_threshold_bytes above. |
| `xla_gpu_reduce_scatter_combine_threshold_bytes` | Integer (bytes) | See xla_gpu_all_reduce_combine_threshold_bytes above. |
| `xla_gpu_enable_pipelined_all_gather` | Boolean (true/false) | Enable pipelinling of all-gather instructions. |
| `xla_gpu_enable_pipelined_reduce_scatter` | Boolean (true/false) | Enable pipelinling of reduce-scatter instructions. |
| `xla_gpu_enable_pipelined_all_reduce` | Boolean (true/false) | Enable pipelinling of all-reduce instructions. |
| `xla_gpu_enable_while_loop_double_buffering` | Boolean (true/false) | Enable double-buffering for while loop. |
| `xla_gpu_enable_all_gather_combine_by_dim` | Boolean (true/false) | Combine all-gather ops with the same gather dimension or irrespective of their dimension. |
| `xla_gpu_enable_reduce_scatter_combine_by_dim` | Boolean (true/false) | Combine reduce-scatter ops with the same dimension or irrespective of their dimension. |

**Additional reading:**
* [GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html#xla-performance-flags)
