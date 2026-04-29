#!/bin/bash

# Script to sweep LoRA scale for LoraE model
MODEL_DIR="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"
TEST_SIZE=20 # Small test size for quick sweep

for SCALE in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.5; do
    echo ">>> Testing with LoRA Scale: $SCALE"
    python test.py --model_dir "$MODEL_DIR"                    --test_model "$CHECKPOINT"                    --robot_visible True                    --lora_scale $SCALE                    --exp_id "Scale_Sweep_$SCALE"                    --test_size $TEST_SIZE
done

# Aggregate after sweep
python aggregate_results.py
