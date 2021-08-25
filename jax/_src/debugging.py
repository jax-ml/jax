# Copyright 2021 Google LLC
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

import jax


def make_hlo(f, optimize=False, metadata=False, platform=None):
  """Utility function for printing JAX-emitted HLO and XLA-compiled HLO.

  Args:
    f: jax function to return hlo for.
    optimize: bool: whether to return platform-specific, XLA-optimized HLO
    metadata: bool: whether to include JAX metadata information
    platform: Optional[str]: None, 'cpu','gpu','tpu' - platform to compile for,
      None uses default.

  Returns:
    wrapped_f: wrapped function which will return HLO in text format.

  Examples:
    Let's define a simple function for which we will generate the HLO:

    >>> def mul_and_add(x, y, z):
    ...   return (x * y) + (x * z)

    First we compute the non-optimized HLO for this function. Notice that the
    results represents the exact sequence of operations written in the function:
    two multiplies followed by an addition:

    >>> print(make_hlo(mul_and_add)(2, 3, 4))
    HloModule xla_computation_mul_and_add.9
    <BLANKLINE>
    ENTRY xla_computation_mul_and_add.9 {
      constant.4 = pred[] constant(false)
      parameter.1 = s32[] parameter(0)
      parameter.2 = s32[] parameter(1)
      multiply.5 = s32[] multiply(parameter.1, parameter.2)
      parameter.3 = s32[] parameter(2)
      multiply.6 = s32[] multiply(parameter.1, parameter.3)
      add.7 = s32[] add(multiply.5, multiply.6)
      ROOT tuple.8 = (s32[]) tuple(add.7)
    }

    Next we compute the optimized HLO for the same function. XLA utilizes the
    distributive property to perform the same computation with a single multiply;
    that is, it is representing the optimized computation ``x * (y + z)``:

    >>> print(make_hlo(mul_and_add, optimize=True, platform='cpu')(2, 3, 4))
    HloModule xla_computation_mul_and_add__1.9
    <BLANKLINE>
    fused_computation {
      param_1.1 = s32[] parameter(1)
      param_2 = s32[] parameter(2)
      add.1 = s32[] add(param_1.1, param_2)
      param_0.1 = s32[] parameter(0)
      ROOT multiply.1 = s32[] multiply(add.1, param_0.1)
    }
    <BLANKLINE>
    ENTRY xla_computation_mul_and_add__1.9 {
      parameter.1 = s32[] parameter(0)
      parameter.2 = s32[] parameter(1)
      parameter.3 = s32[] parameter(2)
      fusion = s32[] fusion(parameter.1, parameter.2, parameter.3), kind=kLoop, calls=fused_computation
      ROOT tuple.8 = (s32[]) tuple(fusion)
    }
  """
  client = jax.lib.xla_bridge.get_backend(platform)
  print_opts = jax.lib.xla_client._xla.HloPrintOptions.short_parsable()
  print_opts.print_metadata = metadata
  def wrapped_fn(*args, **kwargs):
    c = jax.xla_computation(f)(*args, **kwargs)
    if optimize:
      out = client.compile(c).hlo_modules()[0].to_string(print_opts)
    else:
      out = c.as_hlo_module().to_string(print_opts)
    return out.strip()
  return wrapped_fn
