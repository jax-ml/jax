# Persistent Compilation Cache

JAX has an optional disk cache for compiled programs. If enabled, JAX will
store copies of compiled programs on disk, which can save recompilation time
when running the same or similar tasks repeatedly.

## Usage

The compilation cache is enabled when the
[cache-location](https://github.com/google/jax/blob/jax-v0.4.26/jax/_src/config.py#L1206)
is set. This should be done prior to the first compilation. Set the location as
follows:

```python
import jax

# Make sure this is called before jax runs any operations!
jax.config.update("jax_compilation_cache_dir", "cache-location")
```

See the sections below for more detail on `cache-location`.

[`set_cache_dir()`](https://github.com/google/jax/blob/jax-v0.4.26/jax/experimental/compilation_cache/compilation_cache.py#L18)
is an alternate way of setting `cache-location`.

### Local filesystem

`cache-location` can be a directory on the local filesystem. For example:

```python
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
```

Note: the cache does not have an eviction mechanism implemented. If the
cache-location is a directory in the local filesystem, its size will continue
to grow unless files are manually deleted.

### Google Cloud

When running on Google Cloud, the compilation cache can be placed on a Google
Cloud Storage (GCS) bucket. We recommend the following configuration:

*  Create the bucket in the same region as where the workload will run.

*  Create the bucket in the same project as the workload’s VM(s). Ensure that
   permissions are set so that the VM(s) can write to the bucket.

*  There is no need for replication for smaller workloads. Larger workloads
   could benefit from replication.

*  Use “Standard” for the default storage class for the bucket.

*  Set the soft delete policy to its shortest: 7 days.

*  Set the object lifecycle to the expected duration of the workload run.
   For example, if the workload is expected to run for 10 days, set the object
   lifecycle to 10 days. That should cover restarts that occur during the entire
   run. Use `age` for the lifecycle condition and `Delete` for the action. See
   [Object Lifecycle Management](https://cloud.google.com/storage/docs/lifecycle)
   for details. If the object lifecycle is not set, the cache will continue to
   grow since there is no eviction mechanism implemented.

*  All encryption policies are supported.

Assuming that `gs://jax-cache` is the GCS bucket, set `cache-location` as
follows:

```python
import jax

jax.config.update("jax_compilation_cache_dir", "gs://jax-cache")
```

## How it works

The JAX compilation cache works by hashing a number of parameters to create a signature for a compiled function these are:

* The computation performed by the function captured by the non-optimized HLO of the JAX function being hashed
* The Jaxlib version 
* Relevant XLA compilation flags
* Device configuration captured in general, by the number of devices and the topology of the devices. Currently for GPUs, the topology only contains a string representation of the GPU name
* Compression algorithm used to compress the compiled executable
* Any custom hooks 

When the signature for a function created using the parameters above matches that of a compiled function in the persistent cache, the function will not be compiled but just read and deserialized from the persistent cache. Below we outline some observed behavior of the persistent compilation cache in a variety of different runtimes as it relates to which processes write to the cache.

### Single node with single process and single device

This is the simplest runtime and the only process that compiles and writes to the compilation cache is the singular process. The number of devices does not matter to the cache key in this setup, only the type of device does. 

### Single node with single process and multiple devices

Once again the only process that compiles and writes to the compilation cache is the singular proess. The difference between this setup and the previous is that now the number of devices matters in addition to the type of device. 

### Multiple process and multiple devices (on either single or multiple nodes)

In this runtime the first time a program is run (the persistent cache is cold / empty) all processes will compile, but only the process with rank 0 in the global communication group will write to the persistent cache. In subsequent runs, all processes will attempt to read from the persistent cache, so it is important for the persistent cache to be in a shared file system (eg: NFS) or remote storage (eg: GFS). If the persistent cache is local to rank 0, then all processes except rank 0 will once again compile in subsequent runs as a result of a compilation cache miss. 

## Logging cache activity

It can be helpful to examine what exactly is happening with the persistent compilation cache for debugging. While there is no singular canonical way of debugging and examining what's happening in the compilation cache, here are a few suggestions on how to begin.

### Examining cache misses

To merely examine and understand why there are cache misses JAX includes a configuration flag that enables the logging of all cache misses (including persistent compilation cache misses) with their explanations. This can be enabled by setting the following configuration.

```python
import jax

jax.config.update("jax_explain_cache_misses", True)
```

## Pitfalls

There are a couple of pitfalls that have currently been discovered:

* Currently the persistent cache doesn't work with function that have host callbacks. In this situation, caching in completely avoided. 
  - This is because the HLO contains a pointer to the callback and changes from run to run even if the computation and compute infrastructure is exactly the same. 

* Currently the persistent cache doesn't work with a function that uses primitives that implement their own custom_partitioning. 
  - The HLO of the function contains a pointer to the custom_partitioning callback, and leads to different cache keys for the same computation across runs. 
  - In this situation, caching still proceeds, but a different key is produced every time, making the cache ineffective. 


