#!/bin/bash

# Script to collect discrepancy scores in a mixed 50/50 scenario and analyze their distribution

ADAPTIVE_MODEL="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"
TEST_SIZE=10 # Number of episodes to collect enough data points

SCENARIO="seperate_mixed_5050"
BEHAVIOUR="adaptive"

echo "=========================================================="
echo "COLLECTING DISCREPANCY DATA (Scenario: $SCENARIO)"
echo "=========================================================="

# Run the test script to populate the JSON data file
# We don't need visualization for data collection
python test.py --model_dir "$ADAPTIVE_MODEL" --test_model "$CHECKPOINT" \
    --adaptive_lora_scenario "$SCENARIO" \
    --lora_behaviour "$BEHAVIOUR" \
    --test_size $TEST_SIZE "$@"

echo -e "\n=========================================================="
echo "RUNNING STATISTICAL ANALYSIS"
echo "=========================================================="

# Run the analysis python script
python analyze_discrepancy.py
