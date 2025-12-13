# Copyright 2025 The JAX Authors.
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

r"""Tool to reduce the size of a JAX repro

"""
import pathlib

from absl import app
from absl import flags

import os

from jax._src.repro import ddmin
from jax._src.repro import reducer


FLAGS = flags.FLAGS

_START_FILE = flags.DEFINE_string(
    "start_file", None, "The repro file to reduce."
)
_START_FILE = flags.DEFINE_string(
    "test_repro_file", None, "The repro file to reduce."
)
_STRATEGY = flags.DEFINE_string(
    "strategy", "",
    "The reduction strategy to use."
)
_CHUNK_SIZE = flags.DEFINE_integer(
    "chunk_size", -1,
    "The size of a chunk of candidates to try to reduce. If -1, then use the default "
    "of half of the number of candidates"
)
_START_INDEX = flags.DEFINE_integer(
    "start_index", 0,
    "The index into the all candidates to start at."
)

strategies = {
  s.name: s for s in [
    reducer.DropFunctionCallsStrategy,
    reducer.DropExpressionsStrategy,
    reducer.FunctionInlineStrategy,
  ]}


def main(*args):
  input_repro_path = FLAGS.start_file
  if not FLAGS.start_file:
    raise ValueError("Must pass --start_file=")
  input_repro_path = pathlib.Path(os.path.abspath(os.path.expanduser(input_repro_path)))
  with open(input_repro_path, "r") as f:
    input_repro_source = f.read()

  if not FLAGS.strategy:
    raise ValueError("Must pass --strategy=")
  if FLAGS.strategy not in strategies:
    raise ValueError(f"Unrecognized --strategy={FLAGS.strategy}. Known values: {list(strategies.keys())}")
  strategy = strategies[FLAGS.strategy]

  test_repro_source = """
from jax._src import config

def test_repro(repro_fun) -> bool:
  with config.enable_checks(True):
    try:
      repro_fun()
    except AssertionError as e:
      if "Argument count mismatch" in str(e):
        return True
      raise
    else:
      return False

"""
  # For Anselm
  test_repro_source = """
from jax._src import config

def test_repro(repro_fun) -> bool:
  with config.enable_checks(True):
    try:
      repro_fun()
    except ValueError as e:
      if "does not have corresponding jaxpr input" in str(e):
        return True
      raise
    else:
      return False

"""
  # For Anselm 12/12/2025
  test_repro_source = """
import re

_expected = re.compile(r"Argument .UndefinedPrimal\\(Ref.* is not a valid JAX type")
from jax._src import config

def test_repro(repro_fun) -> bool:
  with config.enable_checks(True):
    try:
      repro_fun()
    except TypeError as e:
      if _expected.search(str(e)):
        return True
      raise
    else:
      return False

"""
  # if not FLAGS.repro_test_file:
  #   raise ValueError("Must specify a --repro_test_file")
  # with open(FLAGS.repro_test_file, "r") as f:
  compiled = compile(test_repro_source, "<test_repro_file>", "exec")
  custom_namespace = {}
  custom_namespace['__builtins__'] = __builtins__
  exec(compiled, custom_namespace, custom_namespace)
  test_repro_fun = custom_namespace["test_repro"]

  r = reducer.Repro.make(input_repro_source, input_repro_path,
                         test_repro_fun, strategy)
  chunk_size = FLAGS.chunk_size if FLAGS.chunk_size > 0 else (len(r.all_candidates) + 1 ) // 2
  r_min = ddmin.ddmin(r, chunk_size=chunk_size,
                      start_index=FLAGS.start_index)


if __name__ == '__main__':
  app.run(main)
