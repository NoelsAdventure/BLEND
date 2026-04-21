#!/bin/bash

# Script to test the LoraB model with varying LoRA scales (0% to 100%)
# to see if it becomes gradually more conservative.

MODEL_DIR="trained_models/LoraB_visi_invi_alpha_1024"
CHECKPOINT="05207.pt"

# Base experiment ID (timestamp)
BASE_EXP_ID="lorab_gradual_$(date +'%Y%m%d_%H%M%S')"

# Array of scales: 0.0, 0.1, 0.2, ..., 1.0
scales=("0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")

for scale in "${scales[@]}"; do
    echo "=========================================================="
    echo "TESTING LORAB WITH SCALE: $scale"
    echo "=========================================================="
    
    # Run test.py with the specific scale and experiment ID
    # We pass any additional script arguments (like --visualize) using "$@"
    python test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" --lora_scale "$scale" --exp_id "${BASE_EXP_ID}_scale_${scale}" "$@"
    
    echo "Finished test for scale $scale"
    echo ""
done
