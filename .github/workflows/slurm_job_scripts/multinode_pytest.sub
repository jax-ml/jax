#!/bin/bash
#SBATCH -A ci-jax-gpu
#SBATCH -p compute
#SBATCH -N 2                    # number of nodes
#SBATCH -t 00:15:00             # wall time
#SBATCH -J "ci-jax-gpu"         # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=1     # 1 tasks per machine for now
#SBATCH --overcommit            # Needed for pytorch

set -x

# File system and volume glue code
#-------------------------------------------------------------------------------
CONTAINER="nvcr.io/nvidian/jax_t5x:cuda11.4-cudnn8.2-ubuntu20.04-manylinux2014-multipython"
CONTAINER_NAME="multinode_ci_test_container"

BASE_WORKSPACE_DIR=$GITHUB_WORKSPACE
WORKSPACE_DIR=/workspace

MOUNTS="--container-mounts=$BASE_WORKSPACE_DIR:/$WORKSPACE_DIR"

# Since the docker container doesn't contain MLX drivers for IB, following flags
# are needed to make NCCL work with an ethernet setup
# Note:@sudhakarsingh27 This is very specific, need to abstract this out
EXPORTS="--export=ALL,NCCL_SOCKET_IFNAME=enp45s0f0,NCCL_SOCKET_NTHREADS=2,NCCL_NSOCKS_PERTHREAD=2"
#-------------------------------------------------------------------------------

# Setup command to be run before the actual pytest command
read -r -d '' setup_cmd <<EOF
python3.8 -m pip install --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda_releases.html \
&& python3.8 -m pip install git+https://github.com/google/jax \
&& python3.8 -m pip install pytest \
&& python3.8 -m pip install pytest-forked \
&& mkdir -p /workspace/outputs/
EOF

# Main pytest command that runs the tests
read -r -d '' cmd <<EOF
date \
&& python3.8 -m pip  list | grep jax \
&& python3.8 -m pytest --forked -v -s --continue-on-collection-errors \
    --junit-xml=/workspace/outputs/junit_output_\${SLURM_PROCID}.xml \
    /workspace/tests/multiprocess_gpu_test.py
EOF

# create run specific output directory for ease of analysis
OUTPUT_DIR="${BASE_WORKSPACE_DIR}/outputs/"
mkdir -p $OUTPUT_DIR

# redirect both stdout and stderr in the same file for ease of analysis
OUTFILE="${OUTPUT_DIR}/output-%j-%n.txt"

# Run any setup commands before the actual pytest command to make sure
# that the processes are launched together
echo $setup_cmd
srun -o $OUTFILE -e $OUTFILE \
    --container-writable \
    --container-image="$CONTAINER" \
    --container-name=$CONTAINER_NAME \
    $MOUNTS  \
    $EXPORTS \
    bash -c "${setup_cmd}"

# Barrier command
wait

# Run the actual pytest command
echo $cmd
srun -o $OUTFILE -e $OUTFILE \
    --open-mode=append \
    --container-writable \
    --container-image="$CONTAINER" \
    --container-name=$CONTAINER_NAME \
    $MOUNTS  \
    $EXPORTS \
    bash -c "${cmd}"
set +x
