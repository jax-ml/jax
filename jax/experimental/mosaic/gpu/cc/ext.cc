/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <vector>

#include "absl/hash/hash.h"
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "third_party/py/jax/experimental/mosaic/gpu/cc/tiled_layout.h"

namespace nb = nanobind;
namespace mgpu = jax::mosaic::gpu;

NB_MODULE(ext, m) {
  nb::class_<mgpu::Tiling>(m, "Tiling")
      .def(nb::init<std::vector<std::vector<int64_t>>>(), nb::arg("tiles"))
      .def(
          "tile_shape",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& shape) {
            return nb::tuple(nb::cast(self.TileShape(shape)));
          },
          nb::arg("shape"))
      .def(
          "untile_shape",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& shape) {
            return nb::tuple(nb::cast(self.UntileShape(shape)));
          },
          nb::arg("shape"))
      .def(
          "tile_strides",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& strides) {
            return nb::tuple(nb::cast(self.TileStrides(strides)));
          },
          nb::arg("strides"))
      .def(
          "tile_indices",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& indices) {
            return nb::tuple(nb::cast(self.TileIndices(indices)));
          },
          nb::arg("indices"))
      .def(
          "untile_indices",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& indices) {
            return nb::tuple(nb::cast(self.UntileIndices(indices)));
          },
          nb::arg("indices"))
      .def(
          "tile_nested_shape_strides",
          [](const mgpu::Tiling& self,
             const std::vector<std::vector<int64_t>>& shape,
             const std::vector<std::vector<int64_t>>& strides) {
            auto [tiled_shape, tiled_strides] =
                self.TileNestedShapeStrides(shape, strides);
            nb::list shape_list;
            for (const auto& s : tiled_shape) {
              shape_list.append(nb::tuple(nb::cast(s)));
            }
            nb::list strides_list;
            for (const auto& s : tiled_strides) {
              strides_list.append(nb::tuple(nb::cast(s)));
            }
            return nb::make_tuple(nb::tuple(shape_list),
                                  nb::tuple(strides_list));
          },
          nb::arg("shape"), nb::arg("strides"))
      .def(
          "tile_dimension",
          [](const mgpu::Tiling& self, int64_t dim) {
            return nb::tuple(nb::cast(self.TileDimension(dim)));
          },
          nb::arg("dim"))
      .def("remove_dimension", &mgpu::Tiling::RemoveDimension, nb::arg("dim"))
      .def("canonicalize", &mgpu::Tiling::Canonicalize)
      .def_prop_ro("tiles",
                   [](const mgpu::Tiling& self) {
                     nb::list tiles_list;
                     for (const mgpu::Tiling::Tile& tile : self.tiles()) {
                       tiles_list.append(nb::tuple(nb::cast(tile)));
                     }
                     return nb::tuple(tiles_list);
                   })
      .def("__str__", &mgpu::Tiling::ToString)
      .def("__repr__", &mgpu::Tiling::ToString)
      .def(nb::self == nb::self)
      .def("__hash__", [](const mgpu::Tiling& self) {
        return absl::Hash<mgpu::Tiling>{}(self);
      });
}
