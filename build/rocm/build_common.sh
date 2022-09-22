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

# Common Bash functions used by build scripts

die() {
  # Print a message and exit with code 1.
  #
  # Usage: die <error_message>
  #   e.g., die "Something bad happened."

  echo $@
  exit 1
}

realpath() {
  # Get the real path of a file
  # Usage: realpath <file_path>

  if [[ "$#" != "1" ]]; then
    die "realpath: incorrect usage"
  fi

  [[ "$1" = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

to_lower() {
  # Convert the string to lower case
  # Usage: to_lower <string>

 echo "$1" | tr '[:upper:]' '[:lower:]'
}

calc_elapsed_time() {
  # Calculate elapsed time. Takes nanosecond format input of the kind output
  # by date +'%s%N'
  #
  # Usage: calc_elapsed_time <START_TIME> <END_TIME>

  if [[ $# != "2" ]]; then
    die "calc_elapsed_time: incorrect usage"
  fi

  START_TIME=$1
  END_TIME=$2

  if [[ ${START_TIME} == *"N" ]]; then
    # Nanosecond precision not available
    START_TIME=$(echo ${START_TIME} | sed -e 's/N//g')
    END_TIME=$(echo ${END_TIME} | sed -e 's/N//g')
    ELAPSED="$(expr ${END_TIME} - ${START_TIME}) s"
  else
    ELAPSED="$(expr $(expr ${END_TIME} - ${START_TIME}) / 1000000) ms"
  fi

  echo ${ELAPSED}
}


