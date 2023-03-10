# Copyright 2021 The JAX Authors.
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

# flake8: noqa: F401
import jaxlib.mlir.dialects.builtin as builtin
import jaxlib.mlir.dialects.chlo as chlo
import jaxlib.mlir.dialects.mhlo as mhlo
import jaxlib.mlir.dialects.func as func
import jaxlib.mlir.dialects.ml_program as ml_program
import jaxlib.mlir.dialects.sparse_tensor as sparse_tensor

import jaxlib.mlir.dialects.stablehlo as stablehlo

# Alias that is set up to abstract away the transition from MHLO to StableHLO.
import jaxlib.mlir.dialects.stablehlo as hlo
