#!/bin/bash

# Proof of Concept: Adaptive LoRA switching online
# Compares the adaptive model against static baselines (always_on, always_off)

ADAPTIVE_MODEL="trained_models/LoraE_invi_visi_alpha_128"
CHECKPOINT="05207.pt"
TEST_SIZE=1

# Logic to switch between test.py and visualize.py
SCRIPT="test.py"
for arg in "$@"; do
    if [ "$arg" == "--visualize" ]; then
        SCRIPT="visualize.py"
        # When visualizing, we typically want fewer episodes to save time
        TEST_SIZE=2
        break
    fi
done

SCENARIOS=("seperate_all_ignorant" "seperate_all_aware" "seperate_ignorant_to_aware_step25" "seperate_mixed_5050")
BEHAVIOURS=("switching" "always_off" "always_on")

for scenario in "${SCENARIOS[@]}"; do
    echo -e "\n\n=========================================================="
    echo "SCENARIO: $scenario"
    echo "=========================================================="
    
    for behaviour in "${BEHAVIOURS[@]}"; do
        echo ">>> Testing [LoRA Model] with behaviour: $behaviour..."
        python $SCRIPT --model_dir "$ADAPTIVE_MODEL" --test_model "$CHECKPOINT" \
            --adaptive_lora_scenario "$scenario" \
            --lora_behaviour "$behaviour" \
            --test_size $TEST_SIZE "$@"
    done
done

echo -e "\n\n=========================================================="
echo "ALL TESTS DONE. AGGREGATING RESULTS..."
echo "=========================================================="
# You might need to update these scripts if they rely on the old names
python aggregate_results.py
python plot_experiment_results.py
