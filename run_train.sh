#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

# An optional first .sh argument can prepare the launch environment.
# Example: ./run_train.sh path/to/environment.sh --training.steps 10
if [[ "${1:-}" == *.sh ]]; then
    ENVIRONMENT_SCRIPT=$1
    if [[ ! -f "${ENVIRONMENT_SCRIPT}" ]]; then
        echo "Environment script does not exist: ${ENVIRONMENT_SCRIPT}" >&2
        exit 1
    fi
    source "${ENVIRONMENT_SCRIPT}"
    shift
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module "${MODULE}" --config "${CONFIG}" "$@" --comm.mode="${COMM_MODE}" --training.steps 1
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
    torchrun --nnodes="${NNODES}" --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m torchtitan.train --module "${MODULE}" --config "${CONFIG}" "$@"
fi
