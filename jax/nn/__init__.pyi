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

from typing import Any, List, Literal, overload, Sequence

from jax._src.core import AxisName
from jax._src.cudnn.scaled_matmul_stablehlo import BlockScaleConfig
from jax._src.lax.lax import DotDimensionNumbers
from jax._src.typing import Array, ArrayLike, DTypeLike

from jax.nn import initializers as initializers

Axis = int | Sequence[int] | None


def celu(x: ArrayLike, alpha: ArrayLike = ...) -> Array: ...
def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: ArrayLike | None = ...,
    mask: ArrayLike | None = ...,
    *,
    scale: float | None = ...,
    is_causal: bool = ...,
    query_seq_lengths: ArrayLike | None = ...,
    key_value_seq_lengths: ArrayLike | None = ...,
    local_window_size: int | tuple[int, int] | None = ...,
    implementation: Literal['xla', 'cudnn'] | None = ...,
  ) -> Array: ...
def elu(x: ArrayLike, alpha: ArrayLike = ...) -> Array: ...
def gelu(x: ArrayLike, approximate: bool = ...) -> Array: ...
def get_scaled_dot_general_config(
    mode: Literal['nvfp4', 'mxfp8'],
    global_scale: Array | None = ...,
  ) -> BlockScaleConfig: ...
def glu(x: ArrayLike, axis: int = ...) -> Array: ...
def hard_sigmoid(x: ArrayLike) -> Array: ...
def hard_silu(x: ArrayLike) -> Array: ...
def hard_swish(x: ArrayLike) -> Array: ...
def hard_tanh(x: ArrayLike) -> Array: ...
def identity(x: ArrayLike) -> Array: ...
def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = ...) -> Array: ...
def log_sigmoid(x: ArrayLike) -> Array: ...
def log_softmax(
    x: ArrayLike,
    axis: Axis = ...,
    where: ArrayLike | None = ...,
  ) -> Array: ...
@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = ...,
    b: ArrayLike | None = ...,
    keepdims: bool = ...,
    return_sign: Literal[False] = ...,
    where: ArrayLike | None = ...,
  ) -> Array: ...
@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = ...,
    b: ArrayLike | None = ...,
    keepdims: bool = ...,
    *,
    return_sign: Literal[True],
    where: ArrayLike | None = ...,
  ) -> tuple[Array, Array]: ...
@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = ...,
    b: ArrayLike | None = ...,
    keepdims: bool = ...,
    return_sign: bool = ...,
    where: ArrayLike | None = ...,
  ) -> Array | tuple[Array, Array]: ...
def mish(x: ArrayLike) -> Array: ...
def one_hot(
    x: Any,
    num_classes: int,
    *,
    dtype: Any = ...,
    axis: int | AxisName = ...
  ) -> Array: ...
def relu(x: ArrayLike) -> Array: ...
def relu6(x: ArrayLike) -> Array: ...
def scaled_dot_general(
    lhs: ArrayLike, rhs: ArrayLike,
    dimension_numbers: DotDimensionNumbers,
    preferred_element_type: DTypeLike = ...,
    configs: List[BlockScaleConfig] | None = ...,
    implementation: Literal['cudnn'] | None = ...,
  ) -> Array: ...
def scaled_matmul(
    lhs: Array,
    rhs: Array,
    lhs_scales: Array,
    rhs_scales: Array,
    preferred_element_type: DTypeLike = ...,
  ) -> Array: ...
def selu(x: ArrayLike) -> Array: ...
def sigmoid(x: ArrayLike) -> Array: ...
def silu(x: ArrayLike) -> Array: ...
def soft_sign(x: ArrayLike) -> Array: ...
def softmax(
    x: ArrayLike,
    axis: Axis = ...,
    where: ArrayLike | None = ...
  ) -> Array: ...
def softplus(x: ArrayLike) -> Array: ...
def sparse_plus(x: ArrayLike) -> Array: ...
def sparse_sigmoid(x: ArrayLike) -> Array: ...
def squareplus(x: ArrayLike, b: ArrayLike = ...) -> Array: ...
def standardize(
    x: ArrayLike,
    axis: Axis = ...,
    mean: ArrayLike | None = ...,
    variance: ArrayLike | None = ...,
    epsilon: ArrayLike = ...,
    where: ArrayLike | None = ...
  ) -> Array: ...
def swish(x: ArrayLike) -> Array: ...
def tanh(x: ArrayLike, /) -> Array: ...
def log1mexp(x: ArrayLike) -> Array: ...
