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

import contextlib
from typing import Literal

import jax
from jax._src import config

Category = Literal["nan", "divide", "oob"]


def _is_category_disabled(
    category: Category | None,
) -> bool:
  """Check if the error checking behavior for the given category is disabled."""
  if category is None:
    return False
  if category == "nan":
    raise ValueError("nan is deprecated. Use `_set_error_if_nan` instead.")
  if category == "divide":
    return config.error_checking_behavior_divide.value == "ignore"
  if category == "oob":
    return config.error_checking_behavior_oob.value == "ignore"
  raise ValueError(f"Invalid category: {category}")


def _set_error_if_with_category(
    pred: jax.Array,
    /,
    msg: str,
    category: Category | None = None,
) -> None:
  """Set the internal error state if any element of `pred` is `True`.

  This function is similar to :func:`set_error_if`, but it also takes a category
  argument. The category can be "nan", "divide", or "oob". The error checking
  behavior for each category can be configured using
  :func:`set_error_checking_behavior`. If not provided, there will be no
  category.

  This function is intended for use in JAX internal APIs (e.g., `jax.numpy`)
  to perform category-specific runtime checks tied to the operation being
  performed.
  """
  if _is_category_disabled(category):
    return

  # TODO(mattjj): fix the circular import issue.
  from jax._src import error_check as error_check_lib
  error_check_lib.set_error_if(pred, msg)


def _set_error_if_nan(pred: jax.Array, /):
  """Set the internal error state if any element of `pred` is `NaN`.

  This function is disabled if the `jax_error_checking_behavior_nan` flag is
  set to "ignore".
  """
  if config.error_checking_behavior_nan.value == "ignore":
    return

  # TODO(mattjj): fix the circular import issue.
  import jax.numpy as jnp
  if not jnp.issubdtype(pred.dtype, jnp.floating):  # only check floats
    return

  # TODO(mattjj): fix the circular import issue.
  from jax._src import error_check as error_check_lib
  error_check_lib.set_error_if(jnp.isnan(pred), "NaN encountered")


Behavior = Literal["ignore", "raise"]


class error_checking_behavior:
  """A context manager to set the error checking behavior.

  If both `all` and a category are provided, the category will override the
  `all` setting.

  When the error checking behavior is set to "ignore", all errors will be
  ignored. When set to "raise", errors will be detected and recorded, but an
  exception will not be raised immediately. Users must call
  :func:`raise_if_error` to at the end of the computation to raise the
  exception.
  """

  def __init__(
      self,
      *,
      all: Behavior | None = None,
      nan: Behavior | None = None,
      divide: Behavior | None = None,
      oob: Behavior | None = None,
  ) -> None:
    new_settings = {}
    if all is not None:
      new_settings["nan"] = new_settings["divide"] = new_settings["oob"] = all
    if nan is not None:
      new_settings["nan"] = nan
    if divide is not None:
      new_settings["divide"] = divide
    if oob is not None:
      new_settings["oob"] = oob
    self.new_settings = new_settings
    self.stack = contextlib.ExitStack()

  def __enter__(self):
    config_flags = {
        "nan": config.error_checking_behavior_nan,
        "divide": config.error_checking_behavior_divide,
        "oob": config.error_checking_behavior_oob,
    }
    for key, value in self.new_settings.items():
      self.stack.enter_context(config_flags[key](value))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.stack.close()
