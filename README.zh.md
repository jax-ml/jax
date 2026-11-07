<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# 面向大规模可变换数值计算的 Python 库 (JAX)

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**程序变换 (Transformations)**](#程序变换-transformations)
| [**多设备扩展 (Scaling)**](#多设备扩展-scaling)
| [**安装指南 (Install guide)**](#安装指南)
| [**更新日志 (Change logs)**](https://docs.jax.dev/en/latest/changelog.html)
| [**参考文档 (Reference docs)**](https://docs.jax.dev/en/latest/)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 什么是 JAX？

JAX 是一个面向硬件加速芯片的数组计算与程序变换 Python 库，专为高性能数值计算和大规模机器学习设计。

JAX 能够对原生 Python 及 NumPy 函数进行自动微分。它支持穿透循环、分支、递归以及闭包结构求导，并支持计算任意阶导数（高阶导数）。JAX 不仅通过 [`jax.grad`](#使用-grad-进行自动微分) 支持反向模式自动微分（即反向传播），还支持前向模式自动微分，并且两者可以任意阶自由组合嵌套。

JAX 利用 [XLA](https://www.openxla.org/xla) 编译器将您的 NumPy 程序编译并扩展至 TPU、GPU 及其他硬件加速芯片上。您可以使用 [`jax.jit`](#使用-jit-进行编译) 将自己的纯函数（pure functions）进行编译优化。程序编译与自动微分同样可以任意组合。

深入探究便会发现，JAX 的本质是一个用于在大规模计算下进行**可组合函数变换（composable function transformations）**的可扩展系统。

这是一个科研项目，并非 Google 官方商业产品。请做好遇到一些[棘手细节与粗糙边缘（sharp edges）](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)的准备。欢迎大家尝试体验、[反馈 Bug](https://github.com/jax-ml/jax/issues)，并提出您的宝贵建议！

```python
import jax
import jax.numpy as jnp

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # 作为下一层的输入
  return outputs                # 最后一层不加激活函数

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jax.jit(jax.grad(loss))  # 编译后的损失梯度求值函数
perex_grads = jax.jit(jax.vmap(grad_loss, in_axes=(None, 0, 0)))  # 高效的单样本梯度计算
```

### 目录
* [程序变换 (Transformations)](#程序变换-transformations)
* [多设备扩展 (Scaling)](#多设备扩展-scaling)
* [常见注意事项与容易踩坑的细节 (Current gotchas)](#常见注意事项与容易踩坑的细节)
* [安装指南 (Installation)](#安装指南)
* [引用 JAX (Citing JAX)](#引用-jax)
* [参考文档 (Reference documentation)](#参考文档)

## 程序变换 (Transformations)

JAX 的核心是一个用于变换数值函数的可扩展系统。以下是其中最具代表性的三大变换函数：`jax.grad`、`jax.jit` 与 `jax.vmap`。

### 使用 `grad` 进行自动微分

使用 [`jax.grad`](https://docs.jax.dev/en/latest/jax.html#jax.grad) 高效计算反向模式梯度：

```python
import jax
import jax.numpy as jnp

def tanh(x):
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = jax.grad(tanh)
print(grad_tanh(1.0))
# 输出 0.4199743
```

您可以使用 `grad` 计算任意阶导数：

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# 输出 0.62162673
```

您可以自由地在包含原生 Python 控制流的代码中使用微分：

```python
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = jax.grad(abs_val)
print(abs_val_grad(1.0))   # 输出 1.0
print(abs_val_grad(-1.0))  # 输出 -1.0 (abs_val 将被重新求值)
```

详情请参阅 [JAX 自动微分手册 (Autodiff Cookbook)](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) 以及[自动微分参考文档](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)。

### 使用 `jit` 进行编译

利用 XLA 编译器，通过 [`jit`](https://docs.jax.dev/en/latest/jax.html#just-in-time-compilation-jit) 将函数端到端编译优化。它既可以用作 `@jit` 装饰器，也可以作为高阶函数直接调用。

```python
import jax
import jax.numpy as jnp

def slow_f(x):
  # 逐元素算子在算子融合后可获得极大幅度的性能提升
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jax.jit(slow_f)
%timeit -n10 -r3 fast_f(x)
%timeit -n10 -r3 slow_f(x)
```

使用 `jax.jit` 会对函数内可使用的 Python 控制流类型带来一定约束；详情请参阅教程 [JIT 下的控制流与逻辑运算符 (Control Flow and Logical Operators with JIT)](https://docs.jax.dev/en/latest/control-flow.html)。

### 使用 `vmap` 进行自动向量化

[`vmap`](https://docs.jax.dev/en/latest/jax.html#vectorization-vmap) 能够沿数组的特定轴对函数进行映射。但与简单循环调用函数不同的是，它会将循环下推到函数内部的底层原语操作中，例如将一系列矩阵-向量乘法转换为高效的矩阵-矩阵乘法，从而释放极致性能。

使用 `vmap` 可以让您无需在编写算法核心代码时处处手动维护批次（batch）维度：

```python
import jax
import jax.numpy as jnp

def l1_distance(x, y):
  assert x.ndim == y.ndim == 1  # 仅支持 1D 输入
  return jnp.sum(jnp.abs(x - y))

def pairwise_distances(dist1D, xs):
  return jax.vmap(jax.vmap(dist1D, (0, None)), (None, 0))(xs, xs)

xs = jax.random.normal(jax.random.key(0), (100, 3))
dists = pairwise_distances(l1_distance, xs)
dists.shape  # (100, 100)
```

通过将 `jax.vmap` 与 `jax.grad` 以及 `jax.jit` 组合，我们可以高效计算雅可比矩阵（Jacobian matrices）或单样本独立梯度（per-example gradients）：

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## 多设备扩展 (Scaling)

要将计算任务扩展至成千上万个硬件设备上，您可以组合使用以下任意模式：
* [**基于编译器的自动并行化 (Compiler-based automatic parallelization)**](https://docs.jax.dev/en/latest/parallel.html)：在此模式下，您编写代码时如同面对一台全局单机，编译器会根据用户提供的约束自动决定如何切分数据与划分计算；
* [**显式分片与自动分区 (Explicit sharding and automatic partitioning)**](https://docs.jax.dev/en/latest/parallel.html)：在此模式下，您依然保有全局视角，但数据分片会在 JAX 类型中显式声明，并可通过 `jax.typeof` 进行检查；
* [**单设备视角手动编程 (Manual per-device programming)**](https://docs.jax.dev/en/latest/notebooks/shard_map.html)：在此模式下，您具备针对每个设备的局部数据与计算视角，并通过显式的集合通信原语（collectives）进行交互。

| 模式 | 视角 | 显式分片 (Explicit sharding)? | 显式集合通信 (Explicit Collectives)? |
|---|---|---|---|
| 自动 (Auto) | 全局 (Global) | ❌ | ❌ |
| 显式 (Explicit) | 全局 (Global) | ✅ | ❌ |
| 手动 (Manual) | 单设备 (Per-device) | ✅ | ✅ |

```python
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
mesh = jax.make_mesh((8,), ('data',), axis_types=(AxisType.Explicit,))
set_mesh(mesh)

# 为 FSDP（全分片数据并行）切分模型参数：
for W, b in params:
  print(f'{jax.typeof(W)}')  # f32[512@data,512]
  print(f'{jax.typeof(b)}')  # f32[512]

# 对数据进行分片以实现批次并行：
inputs, targets = jax.device_put((inputs, targets), P('data'))

# 计算梯度，全程自动并行化！
gradfun = jax.jit(jax.grad(loss))
param_grads = gradfun(params, (inputs, targets))
```

详情请参阅[并行化教程](https://docs.jax.dev/en/latest/parallel.html)与[进阶指南](https://docs.jax.dev/en/latest/advanced_guide.html)。

## 常见注意事项与容易踩坑的细节

请参阅 [常见注意事项 Notebook (Gotchas Notebook)](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)。

## 安装指南

### 支持的运行平台

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | 支持 (yes)   | 支持 (yes)    | 支持 (yes)   | 支持 (yes)     | 支持 (yes)          |
| NVIDIA GPU | 支持 (yes)   | 支持 (yes)    | 不适用 (n/a) | 不支持 (no)    | 实验性 (experimental)|
| Google TPU | 支持 (yes)   | 不适用 (n/a)  | 不适用 (n/a) | 不适用 (n/a)   | 不适用 (n/a)        |
| AMD GPU    | 支持 (yes)   | 不支持 (no)   | 不适用 (n/a) | 不支持 (no)    | 实验性 (experimental)|
| Apple GPU  | 不适用 (n/a) | 不支持 (no)   | 实验性 (exp) | 不适用 (n/a)   | 不适用 (n/a)        |
| Intel GPU  | 实验性 (exp) | 不适用 (n/a)  | 不适用 (n/a) | 不支持 (no)    | 不支持 (no)         |

### 安装命令

| 平台 (Platform)  | 安装命令 (Instructions)                                                         |
|-----------------|---------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda13]"`                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                     |
| AMD GPU (Linux) | `pip install -U "jax[rocm7-local]"`                                             |
| Intel GPU       | `pip install -U "jax[oneapi]"`                                                  |

关于其他安装方式，请参阅[官方安装文档](https://docs.jax.dev/en/latest/installation.html)。内容包括从源码编译安装、使用 Docker 容器安装、使用其他特定版本的 CUDA、社区维护的 conda 构建包，以及常见问题解答。

## 引用 JAX

如需在学术研究中引用此仓库，请使用以下 BibTeX 格式：

```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Yash Katariya and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

在上述 BibTeX 条目中，作者姓名按字母顺序列出，版本号旨在对应 [jax/version.py](../main/jax/version.py) 中的版本，年份对应项目的开源发布年份。

JAX 的早期版本（当时仅支持自动微分与 XLA 编译）曾发表于 [SysML 2018 会议论文](https://mlsys.org/Conferences/2019/doc/2018/146.pdf)。目前我们正在撰写一篇更加全面、最新的论文，以系统性阐述 JAX 的核心理念与现代功能。

## 参考文档

有关 JAX API 的详细信息，请参阅 [JAX 官方参考文档 (Reference documentation)](https://docs.jax.dev/)。

若您希望作为 JAX 开发者参与项目贡献，请参阅 [开发者文档 (Developer documentation)](https://docs.jax.dev/en/latest/developer.html)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月15日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
