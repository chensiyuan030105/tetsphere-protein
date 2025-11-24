#!/usr/bin/env bash
set -euo pipefail

# Directory containing all PDB config files
CONFIG_DIR="./config/test"

# Available GPU IDs (modify this if you want a different set)
GPU_IDS=(0)
NUM_GPUS=${#GPU_IDS[@]}

echo "[INFO] Using GPUs: ${GPU_IDS[*]}"
echo "[INFO] Config directory: ${CONFIG_DIR}"

job_index=0  # counts how many training jobs have been launched

for cfg in "$CONFIG_DIR"/*.yaml; do
    # If no files match, skip (avoid literal *.yaml)
    [ -e "$cfg" ] || continue

    echo "============================================================"
    echo "[INFO] Processing config: $cfg"
    echo "============================================================"

    # -----------------------------
    # Step 1: data preprocessing
    # -----------------------------
    # This is mostly CPU-bound; we run it synchronously.
    python data_preprocess/data_processing_carbon.py --config "$cfg"

    # -----------------------------
    # Step 2: training on a GPU
    # -----------------------------
    # Choose a GPU in round-robin fashion
    gpu_id=${GPU_IDS[$(( job_index % NUM_GPUS ))]}
    echo "[INFO] Launching trainer on GPU ${gpu_id} for config: $cfg"

    # Start training in the background on the selected GPU
    CUDA_VISIBLE_DEVICES="${gpu_id}" python trainer.py --config "$cfg" &

    # Increase job index for the next config
    job_index=$(( job_index + 1 ))

    # Limit the number of concurrent training jobs to NUM_GPUS
    # If there are already NUM_GPUS running jobs, wait until at least one finishes.
    while [ "$(jobs -r | wc -l)" -ge "${NUM_GPUS}" ]; do
        sleep 5
    done

    echo
done

# Wait for all background training jobs to finish
wait

echo "[INFO] All configs in $CONFIG_DIR have been processed."

