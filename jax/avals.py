# Copyright 2026 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.avals import (
    AvalDType as AvalDType,
    MeshAxis as MeshAxis,
    bf16 as bf16,
    bool_ as bool,  # noqa: F401
    bool_ as bool_,
    c64 as c64,
    c128 as c128,
    f16 as f16,
    f32 as f32,
    f64 as f64,
    float0 as float0,
    i2 as i2,
    i4 as i4,
    i8 as i8,
    i16 as i16,
    i32 as i32,
    i64 as i64,
    u2 as u2,
    u4 as u4,
    u8 as u8,
    u16 as u16,
    u32 as u32,
    u64 as u64,
)
