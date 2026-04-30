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

from functools import partial
import logging
import os
import re
from typing import Any, Callable

from jax._src import config
from jax._src.repro import ddmin
from jax._src.repro import emitter
from jax._src.repro import reducer


FLAGS = flags.FLAGS

_START_FILE = flags.DEFINE_string(
    "start_file", None, "The repro file to reduce."
)
_STRATEGY = flags.DEFINE_string(
    "strategy", "",
    "The reduction strategy to use."
)
_LIST_STRATEGIES = flags.DEFINE_bool(
    "list_strategies", False,
    "List the available strategies and exit."
)
_VALIDATE_ONLY = flags.DEFINE_bool(
    "validate_only", False,
    "Run only the validation of --start_file, and save a repro for it "
    "then stop."
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
_VALIDATOR = flags.DEFINE_string(
    "validator", None, "The name of the validator module in the validators directory."
)

strategies: dict[str, Callable[[list[Any] | None], emitter.EmitReductionStrategy]] = {
  s.name: s for s in [
    reducer.DropFunctionCallsStrategy,
    reducer.FunctionInlineStrategy,
    reducer.DropExpressionsStrategy,
  ]}


def reduce_repro(*_):
  input_repro_path = FLAGS.start_file
  if not FLAGS.start_file:
    raise ValueError("Must pass --start_file= with the repro to reduce")
  input_repro_path = pathlib.Path(os.path.abspath(os.path.expanduser(input_repro_path)))
  with open(input_repro_path, "r") as f:
    input_repro_source = f.read()

  if FLAGS.list_strategies:
    print("Available strategies:")
    for name, strategy_cls in strategies.items():
      doc = strategy_cls.__doc__ or "No description available."
      lines = doc.strip().split('\n')
      print(f"  * {name}: {lines[0].strip()}")
      for line in lines[1:]:
        print(f"      {line.strip()}")
    return
  if not FLAGS.strategy:
    raise ValueError("Must pass --strategy=  (use --list_strategies for options)")
  if FLAGS.strategy not in strategies:
    raise ValueError(f"Unrecognized --strategy={FLAGS.strategy}. Known values: {list(strategies.keys())}")
  strategy = strategies[FLAGS.strategy]

  if not config.repro_dir.value:
    raise ValueError("Must pass --repro_dir= with the directory to save repros in")

  repro_dir_name = config.repro_dir.value
  if repro_dir_name == "sponge":
    repro_dir_name = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
  repro_dir = pathlib.Path(repro_dir_name)
  if not repro_dir.is_dir():
    raise ValueError(f"repro_dir={repro_dir} is not a directory")
  if not os.access(repro_dir, os.W_OK):
    raise ValueError(f"repro_dir={repro_dir} is not writable")

  if not FLAGS.validator:
    raise ValueError(
        "Must pass --validator= with the name of the module in "
        "repro.validators to use as a repro validator. Alternatively, use "
        "--validator=expected_exception='error message regexp'")

  if FLAGS.validator.startswith("expected_exception="):
    expected_exception_msg = re.compile(
        FLAGS.validator[len("expected_exception="):]
    )
    validate_repro_fun = partial(
        reducer.expected_exception_validate_repro,
        expected_exception_msg_re=expected_exception_msg)
  else:
    raise NotImplementedError(
        f"validator={FLAGS.validator} is not implemented yet")

  r = reducer.Repro.make(input_repro_source, input_repro_path,
                         validate_repro_fun, strategy)
  if FLAGS.validate_only:
    print("Exiting because of --validate_only")
    return
  if len(r.all_candidates) == 0:
    print("There are no reduction candidates")
    return
  logging.info(f"The repro in {FLAGS.start_file} has {len(r.all_candidates)} reduction candidates")
  chunk_size = FLAGS.chunk_size if FLAGS.chunk_size > 0 else (len(r.all_candidates) + 1 ) // 2
  _ = ddmin.ddmin(r, chunk_size=chunk_size, start_index=FLAGS.start_index)


if __name__ == '__main__':
  app.run(reduce_repro)
