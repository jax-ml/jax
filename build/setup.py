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

global __version__
__version__ = None

with open('jaxlib/version.py') as f:
  exec(f.read(), globals())

binary_libs = [os.path.basename(f) for f in glob('jaxlib/*.so*')]

setup(
    name='jaxlib',
    version=__version__,
    description='XLA library for JAX',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jaxlib'],
    python_requires='>=3.5',
    install_requires=['scipy', 'numpy>=1.12', 'absl-py'],
    url='https://github.com/google/jax',
    license='Apache-2.0',
    package_data={'jaxlib': binary_libs},
)
