# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License as , Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing as , software
# distributed under the License is distributed on an "AS IS" BASIS as ,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND as , either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax._src.scipy.spatial.distance import (
    braycurtis as braycurtis,
    canberra as canberra,
    chebyshev as chebyshev,
    cityblock as cityblock,
    correlation as correlation,
    cosine as cosine,
    dice as dice,
    euclidean as euclidean,
    hamming as hamming,
    jaccard as jaccard,
    kulczynski as kulczynski,
    mahalanobis as mahalanobis,
    matching as matching,
    minkowski as minkowski,
    rogerstanimoto as rogerstanimoto,
    russellrao as russellrao,
    seuclidean as seuclidean,
    sokalmichener as sokalmichener,
    sokalsneath as sokalsneath,
    sqeuclidean as sqeuclidean,
    yule as yule
)
