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

from collections.abc import Sequence
import logging
import pathlib

from typing import Any, Callable, Union

from jax._src.repro import tracker
from jax._src.repro.tracker import Call, Statement
from jax._src.repro import emitter
from jax._src.repro import ddmin


ReproFun = Callable[[], Any]  # Reproducers are nullary functions

# Should return `True` if the function does reproduce the failure.
# Otherwise, it may raise an exception or return False.
TestReproFun = Callable[[ReproFun], bool]

# A strategy factory can be called with None to collect the candidates,
# or with a list of candidates to apply those candidates when
# emitting source.
ReductionStrategyFactory = Callable[[list[ddmin.Candidate] | None],
                                     emitter.EmitReductionStrategy]


class DropFunctionCallsStrategy(emitter.EmitReductionStrategy):
  """Drop function calls

  A dropped function call will have the effect that the values it was returning
  are not defined anymore, and will be replaced with `np.ones` of the right shape
  """
  name = "drop_calls"

  def __init__(self, reduce_candidates: Sequence[emitter.Call] | None):
    self.all_candidates: list[emitter.Call] = []
    self.reduce_candidates = reduce_candidates
    self.reduce_candidates_ids = {c.id for c in reduce_candidates} if reduce_candidates else set()

  def keep_call(self, c: Statement) -> bool:
    self.all_candidates.append(c)
    if self.reduce_candidates:
      return (c.id not in self.reduce_candidates_ids)
    return True

### Inlining strategy

class FunctionInlineStrategy(emitter.EmitReductionStrategy):

  name = "inline_calls"

  def __init__(self, reduce_candidates: list[tuple[Statement, bool, int]] | None):
    # Each candidate is a tuple whose first element is a statement.
    # TODO: maybe we can make it a pair, with the second element being an index
    self.all_candidates: list[tuple[Statement, int]] = []
    self.reduce_candidates = reduce_candidates
    self.reduce_candidates_set: set[tuple[int, int]] = {
      (c[0].id, c[1])
      for c in reduce_candidates} if reduce_candidates else set()

  def rewrite_statement(self, s: "Statement") -> tuple[Any, tuple[Any, ...], dict[str, Any], Any]:
    api_name = tracker.func_api_name(s.func)
    if api_name == "jax_cond":
      self.all_candidates.extend([(s, 0), (s, 1)])
      if (s.id, 0) in self.reduce_candidates_set:
        return s.args[1], s.args[3:], s.kwargs, s.result  # type: ignore
      elif (s.id, 1) in self.reduce_candidates_set: # just the false branch
        return s.args[2], s.args[3:], s.kwargs, s.result  # type: ignore
    elif api_name == "jax_jit_call":
      self.all_candidates.append((s, 0))
      if (s.id, 0) in self.reduce_candidates_set:
        return s.args[0], s.args[3:], s.kwargs, s.result  # type: ignore
    elif api_name == "jax_checkpoint_call":
      self.all_candidates.append((s, 0))
      if (s.id, 0) in self.reduce_candidates_set:
        return s.args[0], s.args[3:], s.kwargs, s.result  # type: ignore

    return s.func, s.args, s.kwargs, s.result  # type: ignore


class DropExpressionsStrategy(emitter.EmitReductionStrategy):

  name = "drop_expressions"

  def __init__(self, reduce_candidates: list[tuple[emitter.Call, bool, int]] | None):
    self.all_candidates: list[tuple[emitter.Call, bool, int]] = []
    self.reduce_candidates = reduce_candidates
    self.reduce_candidates_set = {(c[0].id, c[1], c[2])
                                  for c in reduce_candidates} if reduce_candidates else set()

  def keep_expression(self, c: "Call", for_args: bool, atom_idx: int, v: Any) -> bool:
    # Do not drop the args for user functions nor the results for the JAX
    # functions, because they are only used to define values, and if the
    # values are not used then there is no real reduction.
    is_jax = tracker.is_primitive(c.func) or not c.func.is_user
    if is_jax == (not for_args):
      return True
    if self.reduce_candidates:
      return (c.id, for_args, atom_idx) not in self.reduce_candidates_set
    self.all_candidates.append((c, for_args, atom_idx))
    return True


class Repro(ddmin.Repro[ddmin.Candidate]):
  """An implementation of the ddmin.Repro interface for reproducers"""
  def __init__(self,
               repro_source: str,
               repro_path: pathlib.Path,
               test_repro_fun: TestReproFun,
               strategy_factory: ReductionStrategyFactory,
               all_candidates: list[ddmin.Candidate],
               collector: emitter.collector,
               ):
    super().__init__()
    self.repro_path = repro_path
    self.repro_source = repro_source
    self.test_repro_fun = test_repro_fun
    self.all_candidates = all_candidates
    self.collector = collector
    self.strategy_factory = strategy_factory

  def __repr__(self):
    return f"Repro(path={self.repro_path}, nr_candidates={len(self.all_candidates)})"

  def progress_msg(self, state: ddmin.State[ddmin.Candidate]) -> str:
    name = self.strategy_factory.name  # type: ignore
    return f" --start_file={self.repro_path} --strategy={name} --chunk_size={state.chunk_size} --start_index={state.start_index}"

  def get_reduced_repro(self, reduction_candidates: list[ddmin.Candidate]) -> Union["Repro", None]:
    assert reduction_candidates
    emit_strategy = self.strategy_factory(reduction_candidates)
    new_source = self.collector.to_source(strategy=emit_strategy,
                                          repro_name_prefix=None)

    res: str | tuple[emitter.collector, list[ddmin.Candidate]] = \
      Repro.check_repro(new_source, pathlib.Path("<memory>"),
                        self.test_repro_fun, self.strategy_factory, raise_on_failure=False)
    if isinstance(res, str):
      logging.info(f"Not a repro: {res}")
      return None

    collector, all_candidates = res
    repro_path = emitter.save(new_source)
    logging.info(f"** Saved a repro with {len(all_candidates)} candidates at {repro_path}.")

    assert len(all_candidates) < len(self.all_candidates), (len(all_candidates), len(self.all_candidates))
    return Repro(new_source, repro_path, self.test_repro_fun, self.strategy_factory,
                 all_candidates, collector)

  @staticmethod
  def make(repro_source: str,
           repro_path: pathlib.Path,
           test_repro_fun: TestReproFun,
           strategy: ReductionStrategyFactory) -> "Repro":
    res: tuple[emitter.collector, list[ddmin.Candidate]] = \
      Repro.check_repro(repro_source, repro_path, test_repro_fun, strategy,  # type: ignore
                        raise_on_failure=True)
    col, all_candidates = res
    return Repro(repro_source, repro_path, test_repro_fun, strategy, all_candidates, col)

  @staticmethod
  def check_repro(repro_source: str,
                  repro_path: pathlib.Path,
                  test_repro_fun: TestReproFun,
                  strategy_factory: ReductionStrategyFactory, *,
                  raise_on_failure: bool = True) -> str | tuple[emitter.collector, list[ddmin.Candidate]]:
    """Checks if a reproduction source accurately reproduces the target issue.

    This function verifies the reproduction by loading `repro_source` and
    running it through `test_repro_fun`. If successful, it also initializes
    the reduction process by collecting initial reduction candidates.

    Args:
      repro_source: The source code of the reproduction script.
      repro_path: The file path associated with `repro_source`, used for
        error messages.
      test_repro_fun: A predicate function that takes the loaded reproduction
        callable and returns `True` if it reproduces the issue of interest.
      strategy_factory: A factory callable that produces an `EmitReductionStrategy`.
        Used to identify reduction candidates.
      raise_on_failure: Whether to raise a `ValueError` if the source fails to
        reproduce the issue. Defaults to `True`.

    Returns:
      If successful, returns a tuple of `(collector, candidates)`, where:
        * `collector` is an `emitter.collector` wrapping the reproduction.
        * `candidates` is a list of `ddmin.Candidate` objects for reduction.
      If unsuccessful and `raise_on_failure` is `False`, returns an error message string.

    Raises:
      ValueError: If `raise_on_failure` is `True` and the source fails to
        reproduce the issue.
    """
    # While we execute the "main" function from repro_source, collect a repro for it
    collector = emitter.collector(emitter.load(repro_source, repro_path))
    try:
      is_repro = test_repro_fun(collector)
    except Exception as e:
      msg = f"Not a reproducer, it raises: {e} of {type(e)}."
      if raise_on_failure:
        raise ValueError(msg) from e
      return msg
    else:
      if is_repro:
        emit_strategy = strategy_factory(None)
        # Generate source once, to populate the emit_strategy.all_candidates
        new_source = collector.to_source(strategy=emit_strategy,
                                         repro_name_prefix=None)
        # Check the new source to see that it still reproduces, before we
        # start a long reduction process that would not succeed.
        error_msg = (
          "It seemed that the source reproduces the error but the repro collected from it "
          "does not also reproduce the error.")
        try:
          new_main = emitter.load(new_source, pathlib.Path("<memory>"))
          with tracker.flags_override(save_repro_on_uncaught_exception=False):
            new_main_is_repro = test_repro_fun(new_main)
        except Exception as e:
          raise ValueError(f"{error_msg} . Got {e} of type {type(e)}")
        else:
          if not new_main_is_repro:
            raise ValueError(f"{error_msg} . test_repro_fun returns {new_main_is_repro}")

        return collector, emit_strategy.all_candidates  # type: ignore

      msg = f"Not a reproducer, test_repro_fun returns {is_repro}."
      if raise_on_failure:
        raise ValueError(msg)
      return msg
