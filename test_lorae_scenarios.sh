#!/bin/bash

# Script to test LoraE_invi_visi_alpha_128 model in both Visi and Invi scenarios.

MODEL_DIR="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"

# echo "Testing VISIBLE scenario..."
# python test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" --robot_visible True --exp_id "LoraE_Visi" --test_size 1000 "$@"

echo "Testing INVISIBLE scenario..."
# Scale 0 means LoRA is effectively disabled for the invisible scenario
python test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" --robot_visible False --lora_scale 0.0 --exp_id "LoraE_Invi" --test_size 1000 --lora_scale "0.0" "$@"
