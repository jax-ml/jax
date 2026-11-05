# Copyright 2026 The JAX Authors.
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

"""Polynomial approximations for `bessel_i0e` and `bessel_i1e`."""

from jax._src.lax.lax import (
    convert_element_type,
    full_like,
    polynomial,
    reciprocal,
    rsqrt,
    select,
    select_n,
)
from jax._src.typing import Array
import numpy as np


# Sollya `fpminimax` polynomial coefficients for `bessel_i0e(x)` and
# `bessel_i1e(x)`, with a split leading constant `c0_hi + c0_lo` and ascending
# degree order for `lax.polynomial`.
#
# Domain partition on `z = |x| >= 0`:
#   - Region 0: `z in [0, 0.5]`,      `s = z`
#   - Region 1: `z in (0.5, 2.625]`,  `s = z - 1.5`
#   - Region 2: `z in (2.625, 8.0]`,  `s = z - 5.25`
#   - Region 3: `z in (8.0, inf)`,    `s = 1 / z`
#
# Why these intervals, shifts, and splits are used:
#   - On `[0, 0.5]`, expanding directly in `z` around `0` avoids subtracting a
#     shift near `0` (which in Cephes's `0.5 * |x| - 2` loses 2 ULPs near `0`).
#   - On `(0.5, 2.625]` and `(2.625, 8.0]`, the centers `1.5` (`0x1.8p0`) and
#     `5.25` (`0x1.5p2`) have only 2 and 3 significant bits, so `s = z - c` is
#     bit-exact across each interval while keeping `|s| <= 1.125` and
#     `|s| <= 2.75`.
#   - Splitting `f(s) = c0_hi + (c0_lo + s * Q(s))` removes the 0.5-ULP
#     truncation error of `c0` at the cost of 1 extra addition.
#   - In Region 3, `c0_hi` is rounded to 12 bits in `float32` (26 bits in
#     `float64`), and `r = rsqrt(z)` is split into 12-bit (26-bit) `(r_hi, r_lo)`
#     via Veltkamp splitting. The product `c0_hi * r_hi` then fits in 24 (53)
#     bits and is exact without FMA.
#   - For `bessel_i1e`, Region 0 approximates `I_1(z) * exp(-z) / z` and
#     evaluates `x * (c0_hi + p_tail)` to preserve odd symmetry and `-0.0`.
#     Regions 1 and 2 approximate `I_1(z) * exp(-z)` directly.
#   - All four regions select coefficients into a single `lax.polynomial` call
#     (degree 11 in `float32`, degree 22 in `float64`).
#
# Generator script (using `mpmath` + `/usr/bin/sollya` 8.0):
#
#   prec = 350!;
#   display = hexadecimal!;
#   /* For each region [a, b] centered at c (or u = 1/x on [0, 0.125]):
#      f is I0(x)*exp(-x), I1(x)*exp(-x)/x, I1(x)*exp(-x), or
#      sqrt(1/u)*In(1/u)*exp(-1/u).
#      c0_bits is 24 (f32) / 53 (f64) on Regions 0..2, and 12 (f32) / 26 (f64)
#      on Region 3. fmt is SG (f32) or D (f64). */
#   c0_hi = round(evaluate(f, 0), c0_bits, RN);
#   p = fpminimax(f - c0_hi, [|0, 1, ..., deg|], [|fmt...|], [a - c; b - c],
#                 floating, absolute);
#   print(c0_hi);
#   for i from 0 to deg do print(coeff(p, i));

_I0E32_C0_HI = (
    float.fromhex("0x1p0"),
    float.fromhex("0x1.784084p-2"),
    float.fromhex("0x1.6e42bep-3"),
    float.fromhex("0x1.988p-2"),
)

_I0E32_0_05 = (
    float.fromhex("-0x1.60012p-44"),
    float.fromhex("-0x1p0"),
    float.fromhex("0x1.8p-1"),
    float.fromhex("-0x1.aaaaaap-2"),
    float.fromhex("0x1.75552cp-3"),
    float.fromhex("-0x1.0cc8ap-4"),
    float.fromhex("0x1.48127ap-6"),
    float.fromhex("-0x1.54cf5ep-8"),
    float.fromhex("0x1.e8f762p-11"),
    float.fromhex("0x1.f57f18p-13"),
    float.fromhex("-0x1.9e22d2p-12"),
    float.fromhex("0x1.521746p-13"),
)

_I0E32_05_2625 = (
    float.fromhex("0x1.9a23acp-30"),
    float.fromhex("-0x1.2fe95p-3"),
    float.fromhex("0x1.34c2cep-4"),
    float.fromhex("-0x1.18eb74p-5"),
    float.fromhex("0x1.b58b9ap-7"),
    float.fromhex("-0x1.2462dep-8"),
    float.fromhex("0x1.537a8p-10"),
    float.fromhex("-0x1.5a8844p-12"),
    float.fromhex("0x1.3b7834p-14"),
    float.fromhex("-0x1.0ad946p-16"),
    float.fromhex("0x1.a14f8p-19"),
    float.fromhex("-0x1.9b8c08p-22"),
)

_I0E32_2625_8 = (
    float.fromhex("-0x1.04e78ep-27"),
    float.fromhex("-0x1.284e22p-6"),
    float.fromhex("0x1.6bc2f2p-9"),
    float.fromhex("-0x1.f77eacp-12"),
    float.fromhex("0x1.7260ccp-14"),
    float.fromhex("-0x1.1a2f2p-16"),
    float.fromhex("0x1.ba7616p-19"),
    float.fromhex("-0x1.4b47f4p-21"),
    float.fromhex("0x1.93eaccp-24"),
    float.fromhex("-0x1.2d59dcp-26"),
    float.fromhex("0x1.667bfap-28"),
    float.fromhex("-0x1.96f394p-31"),
)

_I0E32_8_INF = (
    float.fromhex("0x1.14cf2ep-16"),
    float.fromhex("0x1.988468p-5"),
    float.fromhex("0x1.cb77ep-6"),
    float.fromhex("0x1.e64cf6p-6"),
    float.fromhex("0x1.e9d3cap-6"),
    float.fromhex("0x1.5c2da2p-2"),
    float.fromhex("-0x1.ebb116p0"),
    float.fromhex("0x1.11d9a8p3"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
)

_I1E32_C0_HI = (
    float.fromhex("0x1p-1"),
    float.fromhex("0x1.c097b8p-3"),
    float.fromhex("0x1.4938fap-3"),
    float.fromhex("0x1.988p-2"),
)

_I1E32_0_05 = (
    float.fromhex("-0x1.7465c6p-45"),
    float.fromhex("-0x1p-1"),
    float.fromhex("0x1.4p-2"),
    float.fromhex("-0x1.2aaaaap-3"),
    float.fromhex("0x1.bfffaep-5"),
    float.fromhex("-0x1.19915ep-6"),
    float.fromhex("0x1.30295cp-8"),
    float.fromhex("-0x1.133752p-10"),
    float.fromhex("0x1.59c9bap-14"),
    float.fromhex("0x1.9f29f2p-13"),
    float.fromhex("-0x1.b2556ap-13"),
    float.fromhex("0x1.5009bcp-14"),
)

_I1E32_05_2625 = (
    float.fromhex("0x1.dd96e2p-28"),
    float.fromhex("0x1.365faep-9"),
    float.fromhex("-0x1.c27a62p-6"),
    float.fromhex("0x1.394032p-6"),
    float.fromhex("-0x1.255f1ep-7"),
    float.fromhex("0x1.b1a34p-9"),
    float.fromhex("-0x1.0bf852p-10"),
    float.fromhex("0x1.1d8d4cp-12"),
    float.fromhex("-0x1.0b06b6p-14"),
    float.fromhex("0x1.c4f5e4p-17"),
    float.fromhex("-0x1.75c328p-19"),
    float.fromhex("0x1.d243f8p-22"),
)

_I1E32_2625_8 = (
    float.fromhex("0x1.0481eap-29"),
    float.fromhex("-0x1.9abc0ap-7"),
    float.fromhex("0x1.5ddfd4p-10"),
    float.fromhex("-0x1.093a52p-13"),
    float.fromhex("0x1.2488f6p-18"),
    float.fromhex("0x1.5429d2p-19"),
    float.fromhex("-0x1.26822ep-20"),
    float.fromhex("0x1.34ca78p-22"),
    float.fromhex("-0x1.a754a8p-25"),
    float.fromhex("0x1.7a73cap-27"),
    float.fromhex("-0x1.0c3fe4p-28"),
    float.fromhex("0x1.3f808ep-31"),
)

_I1E32_8_INF = (
    float.fromhex("0x1.14cf78p-16"),
    float.fromhex("-0x1.326344p-3"),
    float.fromhex("-0x1.7eec2ep-5"),
    float.fromhex("-0x1.534e02p-5"),
    float.fromhex("-0x1.4fcbd6p-5"),
    float.fromhex("-0x1.8e5bacp-2"),
    float.fromhex("0x1.104df6p1"),
    float.fromhex("-0x1.32208cp3"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
    float.fromhex("0x0p0"),
)

_I0E64_C0_HI = (
    float.fromhex("0x1p0"),
    float.fromhex("0x1.784084195afep-2"),
    float.fromhex("0x1.6e42bdf5ed0ep-3"),
    float.fromhex("0x1.988453p-2"),
)

_I0E64_0_05 = (
    float.fromhex("0x1.852e8f195c5eap-79"),
    float.fromhex("-0x1p0"),
    float.fromhex("0x1.8p-1"),
    float.fromhex("-0x1.aaaaaaaaaaaabp-2"),
    float.fromhex("0x1.75555555555aap-3"),
    float.fromhex("-0x1.0ccccccccf1c6p-4"),
    float.fromhex("0x1.48888889b893ep-6"),
    float.fromhex("-0x1.5ca5cac473847p-8"),
    float.fromhex("0x1.46db86e3294p-10"),
    float.fromhex("-0x1.126a8f93ef0fp-12"),
    float.fromhex("0x1.a252c9d721898p-15"),
    float.fromhex("-0x1.433256f2f24ebp-17"),
    float.fromhex("0x1.c406d9d8e5471p-18"),
    float.fromhex("-0x1.7f8dc4b394213p-16"),
    float.fromhex("0x1.3f29f27bc2ea1p-14"),
    float.fromhex("-0x1.a8690746192bbp-13"),
    float.fromhex("0x1.bc6b4ca18961bp-12"),
    float.fromhex("-0x1.69b153c065173p-11"),
    float.fromhex("0x1.bfe35429bfdd9p-11"),
    float.fromhex("-0x1.977c76756aa5cp-11"),
    float.fromhex("0x1.00b3ed661c0bbp-11"),
    float.fromhex("-0x1.9001d44282778p-13"),
    float.fromhex("0x1.224d49e804db2p-15"),
)

_I0E64_05_2625 = (
    float.fromhex("0x1.0b24c8a4cb76ep-61"),
    float.fromhex("-0x1.2fe94f46a410bp-3"),
    float.fromhex("0x1.34c2cdefe6d9ep-4"),
    float.fromhex("-0x1.18eb98d35f28fp-5"),
    float.fromhex("0x1.b58bb61bed5eep-7"),
    float.fromhex("-0x1.245e14876cf8fp-8"),
    float.fromhex("0x1.5374a876455abp-10"),
    float.fromhex("-0x1.5b134d5176a1cp-12"),
    float.fromhex("0x1.3c5c6cc408c5ep-14"),
    float.fromhex("-0x1.03d2ef21773d6p-16"),
    float.fromhex("0x1.840e39d11efdp-19"),
    float.fromhex("-0x1.098d3d03aa8cdp-21"),
    float.fromhex("0x1.4f488d7be07d5p-24"),
    float.fromhex("-0x1.88b9574522f37p-27"),
    float.fromhex("0x1.acd87490059b3p-30"),
    float.fromhex("-0x1.bc57ea4f0b0fp-33"),
    float.fromhex("0x1.be748fe18e8fbp-36"),
    float.fromhex("-0x1.1d10cb7e06f02p-40"),
    float.fromhex("-0x1.a2f39205b7d49p-41"),
    float.fromhex("-0x1.a0dab2947d373p-41"),
    float.fromhex("0x1.fcdb13b8dedddp-42"),
    float.fromhex("0x1.1ddc11d862c1bp-43"),
    float.fromhex("-0x1.78d5ecbb86ba5p-44"),
)

_I0E64_2625_8 = (
    float.fromhex("-0x1.72cae0f030a24p-57"),
    float.fromhex("-0x1.284e240be66d7p-6"),
    float.fromhex("0x1.6bc08537f4e74p-9"),
    float.fromhex("-0x1.f77c59bd480fap-12"),
    float.fromhex("0x1.72dce9c24a12ap-14"),
    float.fromhex("-0x1.1a6108bca230cp-16"),
    float.fromhex("0x1.b1701d8ae3b38p-19"),
    float.fromhex("-0x1.47a320abca9dfp-21"),
    float.fromhex("0x1.ded2f88fbedecp-24"),
    float.fromhex("-0x1.4ddb704516765p-26"),
    float.fromhex("0x1.b8c06603d94d4p-29"),
    float.fromhex("-0x1.1256d426ff69p-31"),
    float.fromhex("0x1.4198902db0123p-34"),
    float.fromhex("-0x1.6319d5510756cp-37"),
    float.fromhex("0x1.71bae6cf83a1cp-40"),
    float.fromhex("-0x1.6b82ad49d1625p-43"),
    float.fromhex("0x1.52afb9d9e52dfp-46"),
    float.fromhex("-0x1.2cd44be7c82cap-49"),
    float.fromhex("0x1.f5b2d17fa5922p-53"),
    float.fromhex("-0x1.7b26f9de2ecbap-56"),
    float.fromhex("0x1.2e68cd7ae7b88p-59"),
    float.fromhex("-0x1.2bf65b1309b7fp-62"),
    float.fromhex("0x1.761c120832b28p-66"),
)

_I0E64_8_INF = (
    float.fromhex("0x1.ea1b282f655fdp-29"),
    float.fromhex("0x1.9884533d45318p-5"),
    float.fromhex("0x1.cb94dd99be147p-6"),
    float.fromhex("0x1.debb1c05a25c9p-6"),
    float.fromhex("0x1.6e87979c814d4p-5"),
    float.fromhex("0x1.71129060b93f4p-4"),
    float.fromhex("0x1.59b2ad406a0ccp-2"),
    float.fromhex("-0x1.8c98c0374fa5ep3"),
    float.fromhex("0x1.03b50c1e051f2p10"),
    float.fromhex("-0x1.c99a047684ccbp15"),
    float.fromhex("0x1.2c7d39ee0c2c5p21"),
    float.fromhex("-0x1.2ca31dbd7b172p26"),
    float.fromhex("0x1.d164749f96db4p30"),
    float.fromhex("-0x1.190ffd30425c4p35"),
    float.fromhex("0x1.09c03d188e4p39"),
    float.fromhex("-0x1.88c9c78468f4fp42"),
    float.fromhex("0x1.c2c74f0321e39p45"),
    float.fromhex("-0x1.8ca74c5cfc2acp48"),
    float.fromhex("0x1.0611c5ac2a716p51"),
    float.fromhex("-0x1.f69cb7ac7d98ap52"),
    float.fromhex("0x1.49f9fca298ce8p54"),
    float.fromhex("-0x1.093d0731a4546p55"),
    float.fromhex("0x1.89b237d8e436ap54"),
)

_I1E64_C0_HI = (
    float.fromhex("0x1p-1"),
    float.fromhex("0x1.c097b8ec11eb5p-3"),
    float.fromhex("0x1.4938f97470406p-3"),
    float.fromhex("0x1.988453p-2"),
)

_I1E64_0_05 = (
    float.fromhex("0x1.89e5cee8de751p-80"),
    float.fromhex("-0x1p-1"),
    float.fromhex("0x1.4p-2"),
    float.fromhex("-0x1.2aaaaaaaaaaabp-3"),
    float.fromhex("0x1.c0000000000a9p-5"),
    float.fromhex("-0x1.199999999e34ap-6"),
    float.fromhex("0x1.311111136e4d7p-8"),
    float.fromhex("-0x1.228a2970f83f5p-10"),
    float.fromhex("0x1.edeb42b812078p-13"),
    float.fromhex("-0x1.7b3d3655f4cd8p-15"),
    float.fromhex("0x1.0bdfca0a69764p-17"),
    float.fromhex("-0x1.db01469acffa1p-20"),
    float.fromhex("0x1.7dd2275593ce4p-19"),
    float.fromhex("-0x1.7990ccd98993fp-17"),
    float.fromhex("0x1.3c129ec917b52p-15"),
    float.fromhex("-0x1.a4484ff8bbb1dp-14"),
    float.fromhex("0x1.b7fb377e7f572p-13"),
    float.fromhex("-0x1.65fe04306bc6bp-12"),
    float.fromhex("0x1.bb34b1831cd15p-12"),
    float.fromhex("-0x1.9324c067248acp-12"),
    float.fromhex("0x1.fbd6bca2948b4p-13"),
    float.fromhex("-0x1.8b9a0058a39dp-14"),
    float.fromhex("0x1.1f0f159dbb6cp-16"),
)

_I1E64_05_2625 = (
    float.fromhex("-0x1.e4cd3e61eb513p-57"),
    float.fromhex("0x1.365faa50b24c9p-9"),
    float.fromhex("-0x1.c27a5d349f6c2p-6"),
    float.fromhex("0x1.39403a911c608p-6"),
    float.fromhex("-0x1.255f7d36aa383p-7"),
    float.fromhex("0x1.b1a1d05400a75p-9"),
    float.fromhex("-0x1.0bed1ed5c4f3ap-10"),
    float.fromhex("0x1.1da58c328b975p-12"),
    float.fromhex("-0x1.0c3e2e37fd95ap-14"),
    float.fromhex("0x1.c27db592f786fp-17"),
    float.fromhex("-0x1.5635c3fc54c22p-19"),
    float.fromhex("0x1.dabd9e2e5e18fp-22"),
    float.fromhex("-0x1.2f01959fd4844p-24"),
    float.fromhex("0x1.663647fcd4327p-27"),
    float.fromhex("-0x1.8a5a6162215bdp-30"),
    float.fromhex("0x1.9623ecad0b1d7p-33"),
    float.fromhex("-0x1.88c6bacfd94dfp-36"),
    float.fromhex("0x1.69a4d944f6aa7p-39"),
    float.fromhex("-0x1.460e727e16eb7p-42"),
    float.fromhex("0x1.62a099ee75e48p-46"),
    float.fromhex("0x1.dd9dca139fa41p-49"),
    float.fromhex("0x1.01932b75339e6p-49"),
    float.fromhex("-0x1.4419d2c484095p-50"),
)

_I1E64_2625_8 = (
    float.fromhex("-0x1.d5aec6236ece9p-57"),
    float.fromhex("-0x1.9abc057bd2633p-7"),
    float.fromhex("0x1.5de3c721f2dd1p-10"),
    float.fromhex("-0x1.093edff6082fbp-13"),
    float.fromhex("0x1.1e39ed6d61648p-18"),
    float.fromhex("0x1.55986b66eebc6p-19"),
    float.fromhex("-0x1.17dab7765989ep-20"),
    float.fromhex("0x1.2e5faf40ac082p-22"),
    float.fromhex("-0x1.105ac03c5a97cp-24"),
    float.fromhex("0x1.b22a2d18e3299p-27"),
    float.fromhex("-0x1.39aeb671b50fdp-29"),
    float.fromhex("0x1.a01b1787a0cd9p-32"),
    float.fromhex("-0x1.fed99a9b22055p-35"),
    float.fromhex("0x1.23f6e5a756a49p-37"),
    float.fromhex("-0x1.382c505220ba3p-40"),
    float.fromhex("0x1.396059f25af9ap-43"),
    float.fromhex("-0x1.28e7bd085e7dep-46"),
    float.fromhex("0x1.0b673cbca8e43p-49"),
    float.fromhex("-0x1.c2dbd1d7d877ep-53"),
    float.fromhex("0x1.576277a13a2a3p-56"),
    float.fromhex("-0x1.146586e744085p-59"),
    float.fromhex("0x1.150fea98d7d9p-62"),
    float.fromhex("-0x1.5b48371c37e01p-66"),
)

_I1E64_8_INF = (
    float.fromhex("0x1.ea1b285daddb3p-29"),
    float.fromhex("-0x1.32633e6df2f57p-3"),
    float.fromhex("-0x1.7efc0e04ee7bbp-5"),
    float.fromhex("-0x1.4f1c8ee7dae35p-5"),
    float.fromhex("-0x1.d743a66fb551fp-5"),
    float.fromhex("-0x1.c26ae9c4f88a4p-4"),
    float.fromhex("-0x1.a63aaa3c3fe57p-2"),
    float.fromhex("0x1.e3934e5580ebcp3"),
    float.fromhex("-0x1.2fc26f840006fp10"),
    float.fromhex("0x1.050df29a6dce6p16"),
    float.fromhex("-0x1.50e6120cbe374p21"),
    float.fromhex("0x1.4cb047bebf2a7p26"),
    float.fromhex("-0x1.fdcb509480851p30"),
    float.fromhex("0x1.315ee3e87fd83p35"),
    float.fromhex("-0x1.1ecc4ebf0fe0ep39"),
    float.fromhex("0x1.a5826455ee57bp42"),
    float.fromhex("-0x1.e16accff9b901p45"),
    float.fromhex("0x1.a5db3b57a5021p48"),
    float.fromhex("-0x1.15b61e71f04cap51"),
    float.fromhex("0x1.097487a119534p53"),
    float.fromhex("-0x1.5b90be76cbc29p54"),
    float.fromhex("0x1.16aa92251009ap55"),
    float.fromhex("-0x1.9cae0075dde85p54"),
)


def _eval_bessel_poly(
    z: Array,
    c0_hi_table: tuple[float, ...],
    r0: tuple[float, ...],
    r1: tuple[float, ...],
    r2: tuple[float, ...],
    r3: tuple[float, ...],
) -> tuple[Array, Array]:
  """Evaluates the 4-region minimax polynomial for `z = |x| >= 0`."""
  in_r0 = z <= 0.5
  in_r1 = z <= 2.625
  in_r2 = z <= 8.0
  idx = (
      convert_element_type(~in_r0, np.int32)
      + convert_element_type(~in_r1, np.int32)
      + convert_element_type(~in_r2, np.int32)
  )

  # Clamp denominator to 8.0 on z <= 8.0 so z = 0 never divides by zero.
  safe_z = select(in_r2, full_like(z, 8.0), z)
  u = reciprocal(safe_z)
  s = select_n(idx, z, z - 1.5, z - 5.25, u)
  c0_hi = select_n(idx, *(full_like(z, c) for c in c0_hi_table))
  coeffs = [
      select_n(idx, *(full_like(z, c) for c in cs))
      for cs in zip(r0, r1, r2, r3, strict=True)
  ]
  p_tail = polynomial(s, coeffs, unroll=len(coeffs))

  # Veltkamp split of r = rsqrt(safe_z) so c0_hi * r_hi is exact without FMA.
  r = rsqrt(safe_z)
  split_const = 4097.0 if z.dtype == np.float32 else 134217729.0
  t = r * split_const
  r_hi = t - (t - r)
  r_lo = r - r_hi
  y_r3 = (c0_hi * r_hi) + (c0_hi * r_lo + p_tail * r)
  return select(in_r2, c0_hi + p_tail, y_r3), in_r0


def bessel_i0e_impl(x: Array) -> Array:
  """Computes `I_0(x) * exp(-|x|)`."""
  if x.dtype not in (np.float32, np.float64):
    return convert_element_type(
        bessel_i0e_impl(convert_element_type(x, np.float32)), x.dtype
    )
  z = abs(x)
  if x.dtype == np.float32:
    y, _ = _eval_bessel_poly(
        z, _I0E32_C0_HI, _I0E32_0_05, _I0E32_05_2625, _I0E32_2625_8, _I0E32_8_INF
    )
  else:
    y, _ = _eval_bessel_poly(
        z, _I0E64_C0_HI, _I0E64_0_05, _I0E64_05_2625, _I0E64_2625_8, _I0E64_8_INF
    )
  return y


def bessel_i1e_impl(x: Array) -> Array:
  """Computes `I_1(x) * exp(-|x|)`."""
  if x.dtype not in (np.float32, np.float64):
    return convert_element_type(
        bessel_i1e_impl(convert_element_type(x, np.float32)), x.dtype
    )
  z = abs(x)
  if x.dtype == np.float32:
    mag, in_r0 = _eval_bessel_poly(
        z, _I1E32_C0_HI, _I1E32_0_05, _I1E32_05_2625, _I1E32_2625_8, _I1E32_8_INF
    )
  else:
    mag, in_r0 = _eval_bessel_poly(
        z, _I1E64_C0_HI, _I1E64_0_05, _I1E64_05_2625, _I1E64_2625_8, _I1E64_8_INF
    )
  sign_x = select(x < 0.0, full_like(x, -1.0), full_like(x, 1.0))
  return select(in_r0, x * mag, sign_x * mag)
