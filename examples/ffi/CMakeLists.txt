# Copyright 2024 The JAX Authors.
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

cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(JAX_FFI_EXAMPLE_ENABLE_CUDA "Enable CUDA support" OFF)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

set(
  JAX_FFI_EXAMPLE_CPU_PROJECTS
  "rms_norm"
  "cpu_examples"
)

foreach(PROJECT ${JAX_FFI_EXAMPLE_CPU_PROJECTS})
  nanobind_add_module("_${PROJECT}" NB_STATIC "src/jax_ffi_example/${PROJECT}.cc")
  target_include_directories("_${PROJECT}" PUBLIC ${XLA_DIR})
  install(TARGETS "_${PROJECT}" LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()

if(JAX_FFI_EXAMPLE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(_cuda_examples SHARED "src/jax_ffi_example/cuda_examples.cu")
  set_target_properties(_cuda_examples PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                  CUDA_STANDARD 17)
  target_include_directories(_cuda_examples PUBLIC ${XLA_DIR})
  install(TARGETS _cuda_examples LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

  nanobind_add_module(_gpu_examples NB_STATIC "src/jax_ffi_example/gpu_examples.cc")
  target_include_directories(_gpu_examples PUBLIC ${XLA_DIR})
  target_link_libraries(_gpu_examples PRIVATE CUDA::cudart)
  install(TARGETS _gpu_examples LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()
