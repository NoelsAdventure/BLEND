"""
Utility script to visualize model weights (LoRA or Standard) as heatmaps.
Calculates Delta W = B @ A for LoRA, or just plots standard Linear weights.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Try to find LoRALinear, if not define dummy for isinstance checks
try:
    from rl.networks.network_utils import LoRALinear
except ImportError:
    class LoRALinear: pass

from rl.networks.model import Policy

def plot_heatmap(data, title, save_path):
    # Adjust figure size based on matrix shape to keep it readable
    h, w = data.shape
    fig_w = max(10, w / 50)
    fig_h = max(8, h / 50)
    
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Input dimension')
    plt.ylabel('Output dimension')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint .pt file')
    parser.add_argument('--mode', type=str, choices=['aggressive', 'conservative', 'base'], default='aggressive', 
                        help='LoRA branch to visualize (only used if type is lora or both)')
    parser.add_argument('--type', type=str, choices=['lora', 'standard', 'both'], default='lora',
                        help='Which type of weights to visualize')
    args, unknown = parser.parse_known_args()

    # 1. Load Config and Setup Paths
    model_dir = args.model_dir.rstrip('/')
    # Use insert(0) to prioritize local model configs/args
    sys.path.insert(0, model_dir)
    
    from configs.config import Config as LoadedConfig
    config = LoadedConfig()
    
    import importlib.util
    arg_spec = importlib.util.spec_from_file_location("model_arguments", os.path.join(model_dir, 'arguments.py'))
    model_arguments = importlib.util.module_from_spec(arg_spec)
    arg_spec.loader.exec_module(model_arguments)
    
    # Temporarily hide visualize_weights-specific args from model_arguments.get_args()
    orig_argv = sys.argv
    sys.argv = [orig_argv[0]] + unknown
    algo_args = model_arguments.get_args()
    sys.argv = orig_argv

    # 2. Find Checkpoint
    if args.checkpoint:
        # Check both direct path and checkpoints/ subdirectory
        if os.path.exists(os.path.join(model_dir, 'checkpoints', args.checkpoint)):
            ckpt_path = os.path.join(model_dir, 'checkpoints', args.checkpoint)
        else:
            ckpt_path = os.path.join(model_dir, args.checkpoint)
    else:
        import glob
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')), key=os.path.getmtime)
        ckpt_path = ckpts[-1] if ckpts else os.path.join(model_dir, 'best_model', 'PPO.pt')

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    print(f"Visualizing weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # 3. Create Model and Load Weights
    # Dummy obs shape for init
    dummy_obs_shape = {
        'robot_node': torch.zeros(1, 9), # Changed from 7 to 9 to match selfAttn_merge_SRNN
        'spatial_edges': torch.zeros(20, 2),
        'temporal_edges': torch.zeros(1, 2),
        'detected_human_num': torch.zeros(1),
        'visible_masks': torch.zeros(20),
        'aggressiveness_factor': torch.zeros(1),
        'conformity_scores': torch.zeros(20, 5)
    }
    
    class MockSpace:
        def __init__(self, shape): self.shape = shape
    obs_space_dict = {k: MockSpace(v.shape) for k, v in dummy_obs_shape.items()}
    
    class MockAction:
        def __init__(self): 
            self.__class__.__name__ = "Box"
            self.shape = (2,)
    
    model = Policy(obs_space_dict, MockAction(), config, base=config.robot.policy, base_kwargs=algo_args)
    model.load_state_dict(state_dict, strict=False)

    # 4. Extract and Visualize
    output_dir = os.path.join(model_dir, 'weight_visuals', f"{args.type}_{os.path.basename(ckpt_path)}")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for name, module in model.named_modules():
        layer_sanitized = name.replace('.', '_')
        
        # A. LoRA Visualization
        if args.type in ['lora', 'both'] and isinstance(module, LoRALinear):
            # Check if it actually has LoRA attributes
            if hasattr(module, 'lora_A_agg'):
                count += 1
                print(f"Plotting LoRA: {name}")
                
                if args.mode == 'aggressive':
                    A = module.lora_A_agg.detach().numpy()
                    B = module.lora_B_agg.detach().numpy()
                else:
                    A = module.lora_A_cons.detach().numpy()
                    B = module.lora_B_cons.detach().numpy()
                
                delta_W = (B @ A) * module.scaling
                plot_heatmap(A, f"LoRA A (Rank Projection) - {name}", os.path.join(output_dir, f"lora_{layer_sanitized}_A.png"))
                plot_heatmap(B, f"LoRA B (Upsample) - {name}", os.path.join(output_dir, f"lora_{layer_sanitized}_B.png"))
                plot_heatmap(delta_W, f"Effective Delta W - {name}", os.path.join(output_dir, f"lora_{layer_sanitized}_delta.png"))

        # B. Standard Weight Visualization
        if args.type in ['standard', 'both']:
            # Handle both raw nn.Linear and the base_layer inside LoRALinear
            target_linear = None
            if isinstance(module, nn.Linear):
                target_linear = module
            elif isinstance(module, LoRALinear) and hasattr(module, 'base_layer'):
                target_linear = module.base_layer
            
            if target_linear:
                count += 1
                print(f"Plotting Standard Weights: {name}")
                W = target_linear.weight.detach().numpy()
                plot_heatmap(W, f"Standard Weights (Backbone) - {name}", os.path.join(output_dir, f"base_{layer_sanitized}_W.png"))

    print(f"Successfully generated {count} heatmaps in {output_dir}")

if __name__ == "__main__":
    main()
