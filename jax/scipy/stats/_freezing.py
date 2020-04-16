# Copyright 2020 Google LLC
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


import inspect
import functools


_required = type("_required", (object, ), {})()


class FrozenDistribution:
    """Fixed-parameter instantiation of an aribtrary distribution.

    Wraps a collection of functions and binds some of their arguments to emulate
    a fixed-parameter distribution. Note that this class does not expose the
    functional API of the underlying distribution since everything is done at
    run-time. Instead, we expose the wrapped functions in the `__getattr__`.
    """
    def __init__(self, name, raw_funcs, assignment):
        self.name = name
        self.assignment = assignment

        self.funcs = {}
        for raw_func in raw_funcs:
            self.funcs[raw_func.__name__] = functools.partial(raw_func, **assignment)

    def __getattr__(self, attr):
        if attr in self.funcs:
            return self.funcs[attr]
        raise AttributeError(
            f"distribution does not have attribute '{attr}', "
            f"registered attributes are [{self._get_func_str()}]"
        )

    def __eq__(self, other):
        if not isinstance(other, Frozen):
            return False
        return self.name == other.name and self.assignment == other.assignment

    def __str__(self):
        return f"{self.name}({self._get_assignment_str()})"

    def __repr__(self):
        return f"{self.name}({self._get_assignment_str()}, funcs=[{self._get_func_str()}])"

    def _get_func_str(self):
        return ', '.join(self.funcs.keys())

    def _get_assignment_str(self):
        return ", ".join(f"{varname}={varvalue}" for varname, varvalue in self.assignment.items())


class Freezer:
    """Facilitator of frozen distribution generation.

    Keeps track of wrapped functions to pass through to distribution instantiations.
    """
    def __init__(self, name, **default_assignment):
        self.name = name
        self.default_assignment = default_assignment
        self.argnames = default_assignment.keys()
        self.funcs = []

    def wrap(self, f):
        parameters = inspect.signature(f).parameters
        for argname in self.argnames:
            if argname not in parameters:
                raise TypeError(
                    f"wrapped function `{self.name}.{f.__name__}` "
                    f"is missing argument `{argname}`"
                )
        self.funcs.append(f)
        return f

    def __call__(self, **assignment):
        for varname in assignment:
            if varname not in self.argnames:
                raise TypeError(f"got an unexpected keyword argument '{varname}'")

        assignment = {**self.default_assignment, **assignment}
        required = [f"'{name}'" for name in assignment if assignment[name] == _required]
        if required:
            raise TypeError(f"missing required arguments: {', '.join(required)}")

        return FrozenDistribution(self.name, self.funcs, assignment)
