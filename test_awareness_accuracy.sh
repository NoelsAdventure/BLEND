#!/bin/bash

# Script to measure the accuracy of human awareness prediction using discrepancy scores

ADAPTIVE_MODEL="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"
TEST_SIZE=2 # Run a few episodes to get a good sample size

SCENARIO="seperate_mixed_5050"
BEHAVIOUR="adaptive"

# Test a few different thresholds
THRESHOLDS=(0.0 0.02 0.05 0.1 0.2)

for thresh in "${THRESHOLDS[@]}"; do
    echo -e "\n\n=========================================================="
    echo "TESTING DISCREPANCY THRESHOLD: $thresh"
    echo "=========================================================="
    
    python test.py --model_dir "$ADAPTIVE_MODEL" --test_model "$CHECKPOINT" \
        --adaptive_lora_scenario "$SCENARIO" \
        --lora_behaviour "$BEHAVIOUR" \
        --discrepancy_threshold "$thresh" \
        --test_size $TEST_SIZE "$@"
done
