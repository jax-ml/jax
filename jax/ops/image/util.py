# Copyright 2019 Google LLC
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


def get_spatial_dim(num_dims, data_format, i):
    # TODO(gehring): implement this more generally
    if num_dims != 4:
        raise NotImplementedError

    if data_format == "NHWC":
        return i + 1
    elif data_format == "NCHW":
        return i + 2
    else:
        raise ValueError("Data format {} not recognized.".format(data_format))


def get_feature_dim(num_dims, data_format):
    # TODO(gehring): implement this more generally
    if num_dims != 4:
        raise NotImplementedError

    if data_format == "NHWC":
        return 3
    elif data_format == "NCHW":
        return 1
    else:
        raise ValueError("Data format {} not recognized.".format(data_format))