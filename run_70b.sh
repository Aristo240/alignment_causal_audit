#!/bin/bash

# Kill background processes
pkill -f vllm

# Set GPUs (usually 0,1 on a 2-GPU GCP instance)
export CUDA_VISIBLE_DEVICES=0,1

# Increase timeout for 70B loading
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

echo "--- Launching Llama 3.1 70B Audit ---"
python experiments/260123_pilot/run_pilot.py