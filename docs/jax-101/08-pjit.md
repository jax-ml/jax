---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "BmtpaCzZBT7U"}

# Introduction to pjit

This guide explains how to use pjit to compile and automatically partition functions in both single and multi-host environments. `Pjit` is `jit` with `in_axis_resources` and `out_axis_resources` to specify how a function should be partitioned. `Pjit` can be useful when the jitted version of fun would not fit in a single device’s memory, or to speed up fun by running each operation in parallel across multiple devices. `Pjit` enables users to shard computations without rewriting using the SPMD partitioner. The returned function has semantics equivalent to those of `fun`, but is compiled to an XLA computation that runs across multiple devices (e.g. multiple GPUs or multiple TPU cores). 

Two examples are shown in this guide to demonstrate how pjit works in both single and multi-host environments. The identity function (`lambda x: x`) is chosen to better show how other input parameters are used. 


```python
jax.experimental.pjit.pjit(fun, in_axis_resources, out_axis_resources, static_argnums=(), donate_argnums=())
```



+++ {"id": "hATDpVCRBuXg"}

## Background 
The core infrastructure that supports parallel model training is the XLA SPMD partitioner. It takes in an XLA program that represents the complete neural net, as if there is only one giant virtual device. In addition to the program, it also takes in partitioning specifications for both function inputs and outputs. The output of the XLA SPMD partitioner is an identical program for N devices that performs communications between devices through collective operations. The program only compiles once per host. **Pjit is the API exposed for the XLA SPMD partitioner in JAX.**

+++ {"id": "qF01HxLwC-s0"}


+++ {"id": "PNP-0AH-Bz48"}

## How it works:
The partitioning over devices happens automatically based on the propagation of the input partitioning specified in `in_axis_resources` and the output partitioning specified in `out_axis_resources`. The resources specified in those two arguments must refer to `mesh axes`, as defined by the `jax.experimental.maps.mesh()` context manager. Note that the `mesh` definition at `pjit` application time is ignored, and the returned function will use the `mesh` definition available at each call site.

Inputs to a pjit’d function will be automatically partitioned across devices if they’re not already correctly partitioned based on `in_axis_resources`. In some scenarios, ensuring that the inputs are already correctly pre-partitioned can increase performance. For example, if passing the output of one pjit’d function to another pjit’d function (or the same pjit’d function in a loop), make sure the relevant `out_axis_resources` match the corresponding `in_axis_resources`.

+++ {"id": "iZNcsMjoHRtj"}

## Single Host Example

+++ {"id": "ej2PCAVGHcIU"}

In this example, we have 
- `mesh`: a mesh of shape (4, 2) and axes named ‘x’ and ‘y’ respectively. 
- `input data`: an 8 by 2 numpy array. 
- `in_axis_resources`: None. So the (8, 2) input data is replicated across all devices.
- `out_axis_resources`: PartitionSpec('x', 'y'). It specifies that the two dimensions of output data are sharded over `x` and `y` respectively. 
- `function`: `lambda x: x`. It is the identity function.

As a result, the pjit’d function applied with the given mesh replicates the input across all devices based on `in_axis_resources`, and then keeps only what each device should have based on `out_axis_resources`. It effectively shards the data on CPU across all accelerators according to the `PartitionSpec`. 

Each parameter is explained in detail below:

+++ {"id": "S4PO128OIg1p"}

### Setup

+++ {"id": "JIofdJEodwd1"}

This tutorial should be run on TPU with 8 devices. Please [build your own runtime](https://cloud.google.com/tpu/docs/jax-pods) on Google Cloud Platform and [connects to it using Jupyter](https://research.google.com/colaboratory/local-runtimes.html). 

```{code-cell}
---
executionInfo:
  elapsed: 1413
  status: ok
  timestamp: 1630522179190
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: 6z5RePsaC93O
---
import jax
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
import numpy as np
```

+++ {"id": "quolviRCIoZG"}

### Mesh
Mesh is defined in [jax/interpreters/pxla](https://github.com/google/jax/blob/main/jax/interpreters/pxla.py#L1389), and it is a numpy array of jax devices in a multi-dimensional grid, alongside names for the axes of this mesh. It is also called the logical mesh.


+++ {"id": "V3Vnmcb7Jq1L"}

In the example we are working with, the first (vertical) axis is named ‘x’ and has length 4, and the second (horizontal) axis is named ‘y’ and has length 2. If a dimension of data is sharded across an axis, each device has a slice of the size of data.shape[dim] divided by mesh_shape[axis]. If data is replicated across an axis, devices on that axis should have the same data. 

```{code-cell}
---
executionInfo:
  elapsed: 5274
  status: ok
  timestamp: 1630522184517
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: c5Yi57UsIt-x
outputId: c0f12f8b-c5b8-4f2b-b8f7-97628add3356
---
mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
# 'x', 'y' axis names are used here for simplicity
mesh = maps.Mesh(devices, ('x', 'y'))
mesh
```

+++ {"id": "GinMJ1kF7pat"}

The actual TPU physical device mesh are connected by Inter-chip Interconnect (ICI) fabric. The logical mesh with given axis names is created to abstract the physical device mesh because it is easier to reshape the logical mesh based on distributed computation needs. The layout of the logical mesh is different from that of the physical mesh.

For example, we can have a physical mesh of size (4, 4, 4). If the computation requires a 16-way data parallelism and 4-way model parallelism, the physical mesh should be abstracted to a logical mesh with a shape of (16, 4). 


+++ {"id": "-EBn4AXojvbe"}

+++ {"id": "JGNV0XCJKPlN"}

### Input Data
A numpy array of size (8,2)

```{code-cell}
---
executionInfo:
  elapsed: 5
  status: ok
  timestamp: 1630522184595
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: q8fi-h6hIv6L
outputId: f448aa10-7ff2-411d-caba-1fc5d55b1fc1
---
input_data = np.arange(8 * 2).reshape(8, 2)
input_data
```

+++ {"id": "IzM6BGB2KhOK"}

### in_axis_resources & out_axis_resources
Pytree of structure matching that of arguments to fun, with all actual arguments replaced by resource assignment specifications. It is also valid to specify a pytree prefix (e.g. one value in place of a whole subtree), in which case the leaves get broadcast to all values in that subtree.

The valid resource assignment specifications are:
- None, in which case the value will be replicated on all devices 
- PartitionSpec, a tuple of length at most equal to the rank of the partitioned value. Each element can be a None, a mesh axis or a tuple of mesh axes, and specifies the set of resources assigned to partition the value’s dimension matching its position in the spec. More details are discussed in the section below (More information on PartitionSpec). 

The size of every dimension has to be a multiple of the total number of resources assigned to it.

out_axis_resources – Like in_axis_resources, but specifies resource assignment for function outputs.


```{code-cell}
---
executionInfo:
  elapsed: 23
  status: ok
  timestamp: 1630522184674
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: BqyXld4aKcbi
---
in_axis_resources=None
out_axis_resources=PartitionSpec('x', 'y')
```

+++ {"id": "6HbI_Y40MN4X"}

### Putting everything together

```{code-cell}
---
executionInfo:
  elapsed: 352
  status: ok
  timestamp: 1630522185196
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: 5Q8pU7AaMQf0
outputId: e8ac522e-e6d0-44e0-b7a2-24229073fcf7
---
f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', 'y'))
 
# Sends data to accelerators based on partition_spec
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)
```

```{code-cell}
---
executionInfo:
  elapsed: 61
  status: ok
  timestamp: 1630522185310
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: pyw91mLZMVQN
outputId: 3fa304be-215b-4f94-b7fd-4599c2c15fc7
---
data
```

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1630522185449
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: qnVf_OIoMTXz
outputId: b1400c97-ef53-4109-f30a-de33148d0189
---
data.device_buffers
```

+++ {"id": "LgZbCPrrMg54"}

The result after applying the pjit’d function is a  ShardedDeviceArray, and device_buffers show what data each device has.  

+++ {"id": "qSQhEouSMjV2"}

**Result visualization:** 
Each color represents a different device. Since an (8,2) array is partitioned across a (4,2) mesh in both ‘x’ and ‘y’ axes, each accelerator ends up with a (2,1) slice of the data. 

+++ {"id": "NQSi0q8cMlj8"}


+++ {"id": "mDkB4FNBK9ct"}

### More information on PartitionSpec:

PartitionSpec is a named tuple, whose element can be a None, a mesh axis or a tuple of mesh axes. Each element describes which mesh dimension the input’s dimension is partitioned across. For example, `PartitionSpec(“x”, “y”)` is a PartitionSpec where the first dimension of data is sharded across `x` axis of the mesh, and the second dimension is sharded across `y` axis of the mesh. 

+++ {"id": "lciyhDKOyvqr"}

**Examples of other possible PartitionSpecs:**

Reminder: mesh is of shape (4, 2), input data is of shape (8, 2) 

### **- `PartitionSpec(“x”, None)`**

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1630522185565
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: oGT68Er2ye0Y
outputId: 5e46b0ac-3534-4990-e245-7ff80022ff8f
---
f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', None))
 
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers
```

+++ {"id": "Tr7KUJpIL026"}

the first dimension of the input is sharded across `x` axis and the other dimensions are replicated across other axes. `None` can also be omitted in the PartitionSpec. If `out_axis_resources = PartitionSpec(“x”, None)` in the example above, the result visualization will be the following:



+++ {"id": "EwN6tPSLy9Tz"}

### **- `PartitionSpec(“y”, None)`**

```{code-cell}
---
executionInfo:
  elapsed: 56
  status: ok
  timestamp: 1630522185672
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: gpbaqxsAy5h-
outputId: 48bfcd07-7235-46f9-c59f-47d2f1986577
---
f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('y', None))
 
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers
```

+++ {"id": "qui5GaSjYX1V"}

the first dimension of the input is sharded across `y` axis and the other dimensions are replicated across other axes. If `out_axis_resources = PartitionSpec(“y”, None)` in the example above, the result visualization will be the following: 

+++ {"id": "E2pbgVJazQfJ"}

### **- `PartitionSpec((“x”, “y”), None)`**

```{code-cell}
---
executionInfo:
  elapsed: 57
  status: ok
  timestamp: 1630522185793
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: qYBDV0VKzTVp
outputId: d9ad5182-3ceb-403e-8b01-d03c09ff9343
---
f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec(('x', 'y'), None))
 
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers
```

+++ {"id": "N0WBRBOmZQI2"}

the first dimension of the input is sharded across both `x` and `y` axis and the other dimensions are replicated across other axes. We can think of this as stretching the 2D mesh into an 1D mesh and then do the partition. If `out_axis_resources = PartitionSpec((“x”, “y”), None)` in the example above, the result visualization will be the following: 


+++ {"id": "d_Vdh_yh3XhP"}

### **- `PartitionSpec(None, 'y')`**

```{code-cell}
---
executionInfo:
  elapsed: 53
  status: ok
  timestamp: 1630522185897
  user:
    displayName: Tina Jia
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgXfNqufLfWAnNdfo39ijAImYD6M7w-Kcd8_eLdSw=s64
    userId: '14134751146587731278'
  user_tz: 360
id: 2iq22gE53LD4
outputId: 2c0c63f0-3970-48e4-8ffa-ee4e7360e570
---
f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec(None, 'y'))
 
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers
```

+++ {"id": "siqNb-an3eUd"}

the second dimension of the input is sharded over y axis and the first dimensions is replicated across other axes. We can think of this as stretching the 2D mesh into an 1D mesh and then do the partition. If out_axis_resources = PartitionSpec(None, 'y') in the example above, the result visualization will be the following:


+++ {"id": "bILc3_4w3ueA"}

### **- `PartitionSpec(None, 'x')`**

+++ {"id": "NjPu86IE34Xk"}

`pjit` will complain when `out_axis_resources` is set to be `PartitionSpec(None, 'x')`. This is because the second dimension of input data is of size 2, but mesh's `x` dimension has size 4. size 2 can not be sharded over size 4. It is important to note that the size of input data has to be divisible by the size of mesh on corresponding dimensions.

+++ {"id": "0iQTcTM-aCkK"}

## Multiple Host Example

+++ {"id": "mP7VeBP0aeNN"}

The following example is run in parallel on each host in a Cloud TPU v3-32 pod slice (4 hosts 32 devices). ([Set up instruction here](https://cloud.google.com/tpu/docs/jax-pods))

In this example, we have 

- `mesh`: a mesh of shape (16, 2) and axes named ‘x’ and ‘y’ respectively. 
- `input data`: Each host contains a quarter (8, 2) of the input data of size (32, 2).
- `in_axis_resources`: PartitionSpec(('x', 'y'),). This lets `pjit` know that the (32, 2) input data is already split evenly across hosts (done by user). 
- `out_axis_resources`: PartitionSpec('x', 'y'). It specifies that the two dimensions of output data are sharded over `x` and `y` respectively.
- `function`: `lambda x: x`. It is the identity function.

The pjit’d function applied with a given mesh distributes an even slice to each device. It effectively shards the data on hosts across all accelerators based on the PartitionSpec.

+++ {"id": "pinw32PtcD3W"}

### Setup
```python
import jax
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
from jax.interpreters import pxla
import numpy as np
```

+++ {"id": "wRNv_eogcLaZ"}

### Mesh

In this example, there are 4 hosts with 32 devices. These physical devices are abstracted to a logical mesh of size (16, 2). The first (vertical) axis is named ‘x’ and has length 16, and the second (horizontal) axis is named ‘y’ and has length 2. Within the global mesh, the first (4, 2) devices are connected to host 0, the second (4, 2) devices are connected to host 1. The first 8 devices (in the order of left to right, top to bottom) always belong to host 0, and the second 8 devices always belong to host 1 and so on. 


```python
mesh_shape = (16, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x', 'y'))
```

+++ {"id": "UnUZlK_Ect1_"}

### Input data

In a multi-host environment, all the devices connected to one host have to contain a subslice of a single continuous large slice of the data. In JAX SPMD, there are no direct communications between hosts, so hosts only talk to each other via collective communication between devices. As a result, users need to handle distributed data loading on hosts. 

In this example, the input array of size (32,2) is manually split into quarters of size (8,2) along the `x` axis by user and assigned to each host.

```python
if jax.process_index() == 0:
 input_data = np.arange(0,16).reshape(8,2)
elif jax.process_index() == 1:
 input_data = np.arange(16,32).reshape(8,2)
elif jax.process_index() == 2:
 input_data = np.arange(32,48).reshape(8,2)
else:
 input_data = np.arange(48,64).reshape(8,2)
```

+++ {"id": "gTS_bgtkdch1"}

### in_axis_resources & out_axis_resources

- `in_axis_resources`: PartitionSpec(('x', 'y'),). This partitions the first dimension of input data over both `x` and `y` axes. This lets Pjit know that the (32, 2) input data is already split evenly across hosts (done by user). Since input argument dimensions partitioned over multi-process mesh axes should be of size equal to the corresponding local mesh axis size, pjit sends the (8, 2) on each host to its devices based on `in_axis_resources`. Since each host has a logical mesh of size (4, 2) within the entire logical mesh, each device has a (1, 2) slice. 
- `out_axis_resources`: PartitionSpec('x', 'y'). It specifies that the two dimensions of output data are sharded over `x` and `y` respectively, so each device gets a (2,1) slice. 

**Note**: in_axis_resources and out_axis_resources are different. Here, in_axis_resources shards input data's first dimension over both `x` and `y`, whereas out_axis_resources shards input data's first dimension only over `x`.

+++ {"id": "8b4kf6GtgPgD"}

### Putting everything together
```python
f = pjit(
   lambda x: x,
   in_axis_resources=PartitionSpec(('x', 'y'),),
   out_axis_resources=PartitionSpec('x', 'y'))

# Sends data to accelerators based on partition_spec
with maps.mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)
```

+++ {"id": "LtaOndLDg3DK"}

**Result on host 0**
