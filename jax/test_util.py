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

from contextlib import contextmanager
import inspect
import functools
from functools import partial
import re
import os
import textwrap
from typing import Dict, List, Generator, Sequence, Tuple, Union
import unittest
import warnings
import zlib

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import numpy.random as npr

from ._src import api
from . import core
from ._src import dtypes as _dtypes
from . import lax
from ._src.config import flags, bool_env, config
from ._src.util import prod, unzip2
from .tree_util import tree_multimap, tree_all, tree_map, tree_reduce
from .lib import xla_bridge
from .interpreters import xla
from .experimental.maps import mesh


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'jax_test_dut', '',
    enum_values=['', 'cpu', 'gpu', 'tpu'],
    help=
    'Describes the device under test in case special consideration is required.'
)

flags.DEFINE_integer(
  'num_generated_cases',
  int(os.getenv('JAX_NUM_GENERATED_CASES', 10)),
  help='Number of generated cases to test')

flags.DEFINE_integer(
  'max_cases_sampling_retries',
  int(os.getenv('JAX_MAX_CASES_SAMPLING_RETRIES', 100)),
  'Number of times a failed test sample should be retried. '
  'When an unseen case cannot be generated in this many trials, the '
  'sampling process is terminated.'
)

flags.DEFINE_bool(
    'jax_skip_slow_tests',
    bool_env('JAX_SKIP_SLOW_TESTS', False),
    help='Skip tests marked as slow (> 5 sec).'
)
