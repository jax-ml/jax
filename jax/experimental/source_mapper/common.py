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
"""Common utilities for generating source maps."""
import contextlib
import dataclasses
import re
from typing import Any, Protocol, Sequence

from absl import flags
import jax
from jax._src import sourcemap


@dataclasses.dataclass(frozen=True)
class SourceMapDump:
  """A container for a source map and the paired generated code."""
  source_map: sourcemap.SourceMap
  generated_code: str
  pass_name: str


class CompileFn(Protocol):

  def __call__(self, work_dir, fn, f_args, f_kwargs, **kwargs) -> Any:
    ...


class GenerateDumpFn(Protocol):

  def __call__(self, compile_result: Any, **kwargs) -> SourceMapDump:
    ...


@dataclasses.dataclass(frozen=True)
class Pass:
  name: str
  compile_fn: CompileFn
  generate_dump: GenerateDumpFn


_pass_registry = {}


def register_pass(pass_: Pass):
  if pass_.name in _pass_registry:
    raise ValueError(f"Pass {pass_.name} already registered")
  _pass_registry[pass_.name] = pass_


def all_passes() -> Sequence[Pass]:
  return list(_pass_registry.values())


def filter_passes(regex: str) -> Sequence[Pass]:
  """Gets all registered passes whose display name matches the given regex."""
  return [
      pass_
      for pass_ in _pass_registry.values()
      if re.match(regex, pass_.name)
  ]


@contextlib.contextmanager
def flag_env(**kwargs):
  """A context manager for setting and restoring flags."""
  old_flags = {kwarg: getattr(flags.FLAGS, kwarg) for kwarg in kwargs}
  for kwarg, new_value in kwargs.items():
    setattr(flags.FLAGS, kwarg, new_value)
  try:
    yield
  finally:
    for kwarg, old_value in old_flags.items():
      setattr(flags.FLAGS, kwarg, old_value)


def compile_with_env(f, f_args, f_kwargs, env_flags, compiler_flags):
  with flag_env(**env_flags):
    jax.jit(lambda *args, **kwargs: f(*args, **kwargs)).lower(  # pylint: disable=unnecessary-lambda
        *f_args, **f_kwargs
    ).compile(compiler_flags)
