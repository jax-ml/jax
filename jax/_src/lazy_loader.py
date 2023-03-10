# Copyright 2022 The JAX Authors.
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

"""A LazyLoader class."""

import importlib
from typing import Any, Callable, List, Sequence, Tuple


def attach(package_name: str, submodules: Sequence[str]) -> Tuple[
    Callable[[str], Any],
    Callable[[], List[str]],
    List[str],
]:
  """Lazily loads submodules of a package.

  Example use:
  ```
  __getattr__, __dir__, __all__ = lazy_loader.attach(__name__, ["sub1", "sub2"])
  ```
  """

  __all__: List[str] = list(submodules)

  def __getattr__(name: str) -> Any:
    if name in submodules:
      return importlib.import_module(f"{package_name}.{name}")
    raise AttributeError(f"module '{package_name}' has no attribute '{name}")

  def __dir__() -> List[str]:
    return __all__

  return __getattr__, __dir__, __all__
