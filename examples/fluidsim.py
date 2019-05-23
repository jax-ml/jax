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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def project(vx, vy):
  h = 1. / vx.shape[0]
  div = -0.5 * h * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0) +
                    np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))

  p = np.zeros(vx.shape)
  for k in range(10):
    p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
             + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)) / 4.

  vx = vx - 0.5 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / h
  vy = vy - 0.5 * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / h
  return vx, vy

def advect(f, vx, vy):
  rows, cols = f.shape
  cell_ys, cell_xs = np.meshgrid(np.arange(rows), np.arange(cols))
  return linear_interpolate(f, cell_xs - vx, cell_ys - vy)

def linear_interpolate(
