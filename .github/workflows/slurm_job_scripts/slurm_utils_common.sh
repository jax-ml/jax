#! /bin/bash

# Copyright 2022 Google LLC
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

# These utility functions are used to monitor SLURM multi-node jobs

job_exit_code() {
    shopt -s lastpipe

    if [ "$#" -ne 1 ]; then
        exit 1
    fi

    JOBID="$1"

    sacct -j "${JOBID}" -n --format=exitcode | sort -r -u | head -1 | cut -f 1 -d":" | sed 's/ //g'

    exit ${PIPESTATUS[0]}
}

job_state(){
    shopt -s lastpipe

    if [ "$#" -ne 1 ]; then
        exit 1
    fi

    JOBID="$1"

    sacct -j "${JOBID}" --format State --parsable2 --noheader |& head -n 1

    exit ${PIPESTATUS[0]}
}

job_nodes(){
    set -euo pipefail
    shopt -s lastpipe

    if [ "$#" -ne 1 ]; then
        exit 1
    fi

    JOBID="$1"

    sacct -j "${JOBID}"  -X -n --format=nodelist%400 | sed 's/ //g'
}

job_time(){
    set -euo pipefail
    shopt -s lastpipe

    if [ "$#" -ne 1 ]; then
        exit 1
    fi

    JOBID="$1"

    ## Note: using export so that this line doesn't cause the script to immediately exit if the subshell failed when running under set -e
    export WALLTIME=$(sacct -j "${JOBID}" --format ElapsedRaw --parsable2 --noheader | head -n 1)

    echo ${WALLTIME:-unknown}
}

job_wait(){
    set -euo pipefail
    shopt -s lastpipe

    if [ "$#" -ne 1 ]; then
        exit 1
    fi

    echo "checking for jobid $1"
    JOBID="$1"

    while true; do
        export STATE=$(job_state "${JOBID}")
        case "${STATE}" in
            PENDING|RUNNING|REQUEUED)
                sleep 15s
                ;;
            *)
                sleep 30s
                echo "Exiting with SLURM job status '${STATE}'"
                exit 0
                ;;
        esac
    done
}