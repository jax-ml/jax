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

setup(
    name='jax',
    version='0.0',
    description='Differentiate, compile, and transform Numpy code.',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jax', 'jax.lib', 'jax.interpreters', 'jax.numpy', 'jax.scipy',
              'jax.experimental'],
    install_requires=['numpy>=1.12', 'six', 'protobuf', 'absl-py',
                      'opt_einsum'],
    url='https://github.com/google/jax',
    license='Apache-2.0',
)
