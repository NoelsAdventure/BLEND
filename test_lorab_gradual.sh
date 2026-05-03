#!/bin/bash

# Script to test the LoraE model with varying LoRA scales (0.0 to 2.0)
# to see how behaviour scales beyond standard ranges.

MODEL_DIR="trained_models/LoraF_invi_visi_rank_1"
CHECKPOINT="03400.pt"

# Array of scales: 0.0, 0.2, 0.4, ..., 2.0
scales=("1.0" "1.1")
# scales=("1.0")

for scale in "${scales[@]}"; do
    echo "=========================================================="
    echo "TESTING LORAE WITH SCALE: $scale"
    echo "=========================================================="
     
    # Run test.py with the specific scale
    # We pass any additional script arguments (like --visualize) using "$@"
    python3 test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" --lora_scale "$scale" --lora_behaviour "fixed_scale" --test_size "500" "$@"
     
    echo "Finished test for scale $scale"
    echo ""
done
