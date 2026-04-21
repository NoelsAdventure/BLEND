#!/bin/bash

# Script to test the newly created LoraB model with robot invisible setting.
# The robot.visible=False is already set in the model's saved config.

MODEL_DIR="trained_models/LoraB_visi_invi_alpha_1024"
CHECKPOINT="05207.pt"

python test.py --model_dir "$MODEL_DIR" --test_model "$CHECKPOINT" "$@"
