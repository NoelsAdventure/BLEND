import logging
import argparse
import os
import sys
import warnings

# Suppress Matplotlib permission errors and noisy cache warnings
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Suppress Gym and Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from rl.networks.envs import make_vec_envs
from rl.evaluation import evaluate
from rl.networks.model import Policy

from crowd_sim import *

MODEL_NAME = "Ours_GST"
MODEL_INDEX = "05207"

def main():
    """
    The main function for testing a trained model
    """
    parser = argparse.ArgumentParser(description='Parse configuration file')
    parser.add_argument('--model_dir', type=str, default=f'trained_models/{MODEL_NAME}')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--test_case', type=int, default=-1)
    parser.add_argument('--test_model', type=str, default=f'{MODEL_INDEX}.pt')
    parser.add_argument('--render_traj', default=False, action='store_true')
    parser.add_argument('--save_slides', default=False, action='store_true')
    
    # Arguments added for Adaptive LoRA PoC. Keep this set in sync with
    # test.py — they share the same Adaptive-LoRA CLI surface so the same
    # shell scripts (test_adaptive_lora_poc.sh) can drive either entry point.
    parser.add_argument('--lora_scale', type=float, default=1.0)
    parser.add_argument('--lora_behaviour', type=str, choices=['always_off', 'always_on', 'fixed_scale', 'switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'switching_pred', 'adaptive_gt', 'adaptive_action_gt', 'adaptive_fullfinetune_gt', 'fixed_fullfinetune_scale', 'Gensafenav_cons_upcost', 'adaptive_discrepancy', 'adaptive_discrepancynew', 'adaptive_pred', 'none'], default='adaptive_discrepancy')
    parser.add_argument('--discrepancy_threshold', type=float, default=0.05, help='Threshold for classifying a human as aware/friendly based on discrepancy score')
    parser.add_argument("--discrepancy_m", type=int, default=1, help="Number of consecutive times the score must be above threshold to classify as aware")
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--robot_visible', type=str, default=None, help='Override robot visibility: True or False')
    parser.add_argument('--human_num', type=int, default=None, help='Override number of humans')
    parser.add_argument('--robot_v_pref', type=float, default=None, help='Override robot preferred speed (env_config.robot.v_pref).')
    parser.add_argument('--render_only_cases', type=str, default=None,
                        help='Comma-separated test_case indices to render PNGs/MP4 for. '
                             'Other episodes still run (to keep case_counter / RNG advance '
                             'aligned with test.py), but their frames are skipped. Use with '
                             '--test_size = max(list)+1 and DO NOT also pass --test_case.')
    parser.add_argument('--test_size', type=int, default=1, help='Number of episodes to test')
    parser.add_argument('--adaptive_lora_scenario', type=str, choices=['seperate_ignorant_to_aware_step25', 'seperate_mixed_5050', 'seperate_all_ignorant', 'seperate_all_aware', 'cluster_aware_ignorant', 'none'], default='none')
    parser.add_argument('--awareness_eval', type=str,
                        choices=['always', 'pred_only', 'off'], default='always',
                        help='Same flag as test.py — see test.py --help for details.')
    parser.add_argument('--fullfinetune_model_dir', type=str, default='trained_models/Fullfinetune_invi_visi_new',
                        help='Endpoint model directory for adaptive_fullfinetune_gt dense-weight interpolation.')
    parser.add_argument('--fullfinetune_test_model', type=str, default='03400.pt',
                        help='Endpoint checkpoint filename for adaptive_fullfinetune_gt dense-weight interpolation.')
    parser.add_argument('--predictor_tag', type=str, default=None,
                        help='Same flag as test.py — see test.py --help for details.')
    parser.add_argument('--save_episode_dump', type=str, default='auto',
                        choices=['auto', 'always', 'never'],
                        help='Same flag as test.py — see test.py --help for details.')
    # Use parse_known_args to ignore arguments meant for the environment
    test_args, unknown = parser.parse_known_args()
    
    if test_args.save_slides:
        test_args.visualize = True

    import importlib.util
    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]

    # Load arguments from the model directory
    args_file_path = os.path.join(model_dir_temp, 'arguments.py')
    spec = importlib.util.spec_from_file_location("model_arguments", args_file_path)
    model_arguments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arguments)
    get_args = getattr(model_arguments, 'get_args')

    # Temporarily hide test-specific args from get_args()
    orig_argv = sys.argv
    sys.argv = [orig_argv[0]] + unknown
    algo_args = get_args()
    sys.argv = orig_argv

    # Load config from the model directory
    config_file_path = os.path.join(model_dir_temp, 'configs/config.py')
    spec = importlib.util.spec_from_file_location("model_config", config_file_path)
    model_config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_config_mod)
    Config = getattr(model_config_mod, 'Config')

    env_config = config = Config()
    
    # Apply overrides from command line
    if test_args.robot_visible is not None:
        env_config.robot.visible = (test_args.robot_visible.lower() == 'true')
    if test_args.human_num is not None:
        env_config.sim.human_num = test_args.human_num
    if test_args.robot_v_pref is not None:
        env_config.robot.v_pref = test_args.robot_v_pref

    env_config.aci_related.noise_clip_for_conformity_scores = 0.0
    env_config.aci_related.noise_std_for_conformity_scores = 0.0
    env_config.aci_related.noise_std_for_cost = 0.0
    env_config.aci_related.noise_clip_for_cost = 0.0

    # configure logging and device
    log_dir = os.path.join(test_args.model_dir, 'test')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'test_visual.log') if test_args.visualize else os.path.join(log_dir, f'test_{test_args.test_model}_visualize.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda:0" if algo_args.cuda and torch.cuda.is_available() else "cpu")

    # Match test.py's RNG init so --test_case K reproduces test.py's ep K.
    import numpy as _np
    _seed = getattr(algo_args, 'seed', 42)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    _np.random.seed(_seed)
    torch.set_num_threads(1)

    # Visualization setup
    if test_args.visualize:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.ion()
        plt.show()
    else:
        ax = None

    load_path = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model)
    print(f"Loading model from: {load_path}")

    # create an environment
    eval_dir = os.path.join(test_args.model_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
  
    env_config.reward.base_collision_penalty = -20
    env_config.render_traj = test_args.render_traj
    env_config.save_slides = test_args.save_slides
    env_config.save_path = os.path.join(test_args.model_dir, 'social_eval', test_args.test_model[:-3])
    env_config.args = algo_args

    envs = make_vec_envs(algo_args.env_name, algo_args.seed, 1,
                         algo_args.gamma, eval_dir, device, allow_early_resets=True,
                         config=env_config, ax=ax, test_case=test_args.test_case, pretext_wrapper=config.env.use_wrapper)

    if config.robot.policy not in ['orca', 'social_force']:
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            env_config,
            base_kwargs=algo_args,
            base=config.robot.policy)
        
        state_dict = torch.load(load_path, map_location=device)
        
        # LoRA mapping
        if hasattr(env_config, 'lora') and getattr(env_config.lora, 'use_lora', False):
            new_state_dict = {}
            model_state_dict = actor_critic.state_dict()
            for key, value in state_dict.items():
                base_layer_key = key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                if base_layer_key in model_state_dict:
                    new_state_dict[base_layer_key] = value
                else:
                    new_state_dict[key] = value
            actor_critic.load_state_dict(new_state_dict, strict=False)
        else:
            actor_critic.load_state_dict(state_dict)

        actor_critic.base.nenv = 1
        
        # Apply dynamic LoRA scale
        from rl.networks.network_utils import LoRALinear, LoRAAdapter
        for module in actor_critic.modules():
            if isinstance(module, (LoRALinear, LoRAAdapter)):
                module.dynamic_scale = test_args.lora_scale
        
        # Sync with environment for plotting
        if hasattr(envs.venv, 'envs'):
            envs.venv.envs[0].env.robot.lora_scale = test_args.lora_scale
        else:
            envs.venv.unwrapped.envs[0].env.robot.lora_scale = test_args.lora_scale

        nn.DataParallel(actor_critic).to(device)
    else:
        actor_critic = None

    # Setup visualization save path
    scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
    save_path = os.path.join("visualizations", os.path.basename(model_dir_temp), scenario)
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"Videos will be saved to {save_path}")

    # Call evaluate with video saving enabled
    evaluate(actor_critic, envs, 1, device, test_args.test_size, logging, config, algo_args, model_dir_temp, True, test_args, video_save_path=save_path)

if __name__ == '__main__':
    main()
