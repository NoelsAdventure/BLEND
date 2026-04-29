import os
import json
import pandas as pd
from datetime import datetime

def aggregate_all_results(models_base_dir='trained_models', output_dir='result', filter_models=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    all_data = []

    # Decide which models to iterate over
    if filter_models:
        model_names = [m for m in filter_models if os.path.isdir(os.path.join(models_base_dir, m))]
    else:
        model_names = [m for m in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, m))]

    # Iterate through selected model directories
    for model_name in model_names:
        model_path = os.path.join(models_base_dir, model_name)
            
        summary_path = os.path.join(model_path, 'test', 'all_evaluations.json')
        
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    model_evals = json.load(f)
                
                for exp_id, data in model_evals.items():
                    # Get number of episodes (with fallback for older logs)
                    num_ep = data['summary'].get('num_episodes')
                    if num_ep is None:
                        # Fallback: if 'episodes' exists in this entry
                        num_ep = len(data.get('episodes', [])) if 'episodes' in data else 'N/A'

                    # Flatten the nested dictionary for CSV/DataFrame
                    flat_entry = {
                        'model_name': model_name,
                        'exp_id': exp_id,
                        'timestamp': data.get('timestamp', ''),
                        'num_episodes': num_ep,
                        'success_rate': data['summary'].get('success_rate', 0),
                        'collision_rate': data['summary'].get('collision_rate', 0),
                        'timeout_rate': data['summary'].get('timeout_rate', 0),
                        'avg_nav_time': data['summary'].get('avg_nav_time', 0),
                        'avg_path_length': data['summary'].get('avg_path_length', 0),
                        'avg_uncertainty': data['summary'].get('avg_uncertainty', 0),
                        'robot_visible': data['config'].get('robot_visible', 'N/A'),
                        'use_lora': data['config'].get('use_lora', 'N/A'),
                        'lora_scale': data['config'].get('lora_scale', 'N/A'),
                        'adaptive_scenario': data['config'].get('adaptive_lora_scenario', 'none')
                    }
                    all_data.append(flat_entry)
            except Exception as e:
                print(f"Error processing {summary_path}: {e}")

    if not all_data:
        print("No evaluation data found for the specified models.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp (newest first)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=False)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'master_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved master CSV to {csv_path}")

    # Save to JSON
    json_path = os.path.join(output_dir, 'master_summary.json')
    df.to_json(json_path, orient='records', indent=4, date_format='iso')
    print(f"Saved master JSON to {json_path}")

    # Print a quick summary table to console
    print("\n--- Summary Results ---")
    cols_to_show = ['exp_id', 'num_episodes', 'success_rate', 'collision_rate', 'avg_path_length']
    print(df[cols_to_show].head(20).to_string(index=False))

if __name__ == "__main__":
    # Edit this list to see specific models. 
    # Leave it as an empty list [] to aggregate ALL models in trained_models/
    MODELS_TO_AGGREGATE = [
        "LoraE_invi_visi_alpha_128",
        "Ours_GST",
        "ours_gst_visible_seed_42_curr_buffer_0.25_c_l_0.4_clip_param_0.08_considered_steps_2_alpha_0.1_noise_0_0.0"
    ]
    
    aggregate_all_results(filter_models=MODELS_TO_AGGREGATE if MODELS_TO_AGGREGATE else None)
