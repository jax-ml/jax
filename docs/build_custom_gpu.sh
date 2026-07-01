#!/bin/bash
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

python -m pip install pybind11==2.10.1
mkdir -p build
touch build/__init__.py
pybind_include_path=$(python -c "import pybind11; print(pybind11.get_include())")
python_executable=$(python -c 'import sys; print(sys.executable)')
#python_include_path=$(python -c 'from distutils.sysconfig import get_python_inc;print(get_python_inc())')
echo pybind_include_path=$pybind_include_path
echo python_executable=$python_executable

nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3 --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c gpu_ops/rms_norm_kernels.cu -o build/rms_norm_kernels.cu.o
c++ -I/usr/local/cuda/include -I$pybind_include_path $(${python_executable}3-config --cflags) -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -o build/gpu_ops.cpp.o -c gpu_ops/gpu_ops.cpp
c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared  -o build/gpu_ops$(${python_executable}3-config --extension-suffix) build/gpu_ops.cpp.o build/rms_norm_kernels.cu.o -L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
strip build/gpu_ops$(${python_executable}3-config --extension-suffix)
