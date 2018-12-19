# Copyright 2018 Google LLC
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

from setuptools import setup
from glob import glob
import os

binary_libs = [os.path.basename(f) for f in glob('jaxlib/*.so*')]

setup(
    name='jaxlib',
    version='0.1.3',
    description='XLA library for JAX',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jaxlib'],
    install_requires=['numpy>=1.12', 'six', 'protobuf>=3.6.0', 'absl-py'],
    url='https://github.com/google/jax',
    license='Apache-2.0',
    package_data={'jaxlib': binary_libs},
)
