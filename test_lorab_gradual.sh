#!/bin/bash

# Script to test the LoraE model with varying LoRA scales (0.0 to 2.0)
# to see how behaviour scales beyond standard ranges.

MODEL_DIR="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"

# Array of scales: 0.0, 0.2, 0.4, ..., 2.0
scales=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0" "1.2" "1.4" "1.6" "1.8" "2.0")

for scale in "${scales[@]}"; do
    echo "=========================================================="
    echo "TESTING LORAE WITH SCALE: $scale"
    echo "=========================================================="
     
    # Run test.py with the specific scale
    # We pass any additional script arguments (like --visualize) using "$@"
    python test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" --lora_scale "$scale" "$@"
     
    echo "Finished test for scale $scale"
    echo ""
done
