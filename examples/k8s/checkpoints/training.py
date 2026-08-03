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

"""JAX on Kubernetes: Probes and Checkpointing Example.

This script demonstrates a robust distributed training loop that:
1. Signals its health to Kubernetes via a heartbeat file (/tmp/healthy).
2. Uses Orbax to save checkpoints directly to Google Cloud Storage (GCS).
3. Gracefully handles preemption (SIGTERM) to save a final checkpoint.
"""

import os
import signal
import pathlib
import time
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

LIVENESS_FILE = pathlib.Path("/tmp/healthy")

def touch_liveness():
    """Signals to Kubernetes that the training loop is healthy."""
    LIVENESS_FILE.touch()

# 1. Initialize Distributed JAX
# On GKE, this auto-discovers peers via the Kubernetes API
jax.distributed.initialize()
touch_liveness()

def run_training():
    # 2. Initialize State
    # For this example, we use a simple dict. 
    # In a real scenario, this would be a Flax TrainState.
    state = {"params": jnp.ones((10,)).tolist(), "step": 0}
    
    # 3. Configure Orbax Checkpoint Manager with GCS.
    # We use an environment variable for the checkpoint directory to allow
    # easy configuration without changing the code.
    checkpoint_dir = os.environ.get(
        "CHECKPOINT_DIR", "gs://YOUR_BUCKET_NAME/project-alpha/v1/"
    )
    
    # StandardCheckpointer handles basic python trees (lists, dicts, etc)
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    )

    # 4. Restore from latest checkpoint if it exists
    latest_step = mngr.latest_step()
    if latest_step is not None:
        state = mngr.restore(latest_step, items=state)
        print(f"Resumed from step {latest_step}")

    # 5. Graceful Preemption Handling
    exit_now = False
    def sigterm_handler(signum, frame):
        nonlocal exit_now
        print("Received SIGTERM! Saving emergency checkpoint...")
        exit_now = True

    signal.signal(signal.SIGTERM, sigterm_handler)

    # 6. Training Loop
    start_step = int(state["step"])
    print(f"Starting training loop from step {start_step}...")
    
    for step in range(start_step, 10000):
        # Simulate a training step
        time.sleep(1) 
        state["step"] = step
        
        # Periodic Save (every 10 steps for this example)
        if step % 10 == 0:
            # Synchronous save for simplicity in this example
            mngr.save(step, items=state)
            print(f"Saved checkpoint at step {step}")

        # Kubernetes Health Signaling (heartbeat)
        touch_liveness()

        # Emergency Save & Exit on Preemption
        if exit_now:
            mngr.save(step, items=state, force=True)
            mngr.wait_until_finished()
            print(f"Graceful exit at step {step}")
            break

    # 7. Final Cleanup
    jax.distributed.shutdown()

if __name__ == "__main__":
    run_training()
