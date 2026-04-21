import logging
import argparse
import os
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

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
    # the following parameters will be determined for each test run
    parser = argparse.ArgumentParser('Parse configuration file')
    # the model directory that we are testing
    parser.add_argument('--model_dir', type=str, default=f'trained_models/{MODEL_NAME}')
    # render the environment or not
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--test_case', type=int, default=-1)
    # model weight file you want to test
    parser.add_argument('--test_model', type=str, default=f'{MODEL_INDEX}.pt')
    # whether to save trajectories of episodes
    parser.add_argument('--render_traj', default=False, action='store_true')
    # whether to save slide show of episodes
    parser.add_argument('--save_slides', default=False, action='store_true')
    # dynamic LoRA scale for testing gradual changes
    parser.add_argument('--lora_scale', type=float, default=1.0)
    parser.add_argument('--exp_id', type=str, default=None)
    
    # Use parse_known_args so test.py only takes what it needs
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
    import sys
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
    env_config.aci_related.noise_clip_for_conformity_scores = 0.0
    env_config.aci_related.noise_std_for_conformity_scores = 0.0
    
    env_config.aci_related.noise_std_for_cost = 0.0
    env_config.aci_related.noise_clip_for_cost = 0.0
    # configure logging and device
    # print test result in log file
    log_file = os.path.join(test_args.model_dir,'test')
    if not os.path.exists(log_file):
        print(f"log_file: {log_file}")
        os.mkdir(log_file)
    if test_args.visualize:
        log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
    else:
        log_file = os.path.join(test_args.model_dir, 'test', 'test_' + test_args.test_model + '.log')



    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('robot FOV %f', config.robot.FOV)
    logging.info('humans FOV %f', config.humans.FOV)

    current_seed = algo_args.seed
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


    torch.set_num_threads(1)
    device = torch.device("cuda" if algo_args.cuda else "cpu")

    logging.info('Create other envs with new settings')

    # set up visualization
    if test_args.visualize:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6.5, 6.5) # 6
        ax.set_ylim(-6.5, 6.5)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # ax.set_xlabel('x(m)', fontsize=16)
        # ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()
    else:
        ax = None


    load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
    print(load_path)

    # create an environment
    env_name = algo_args.env_name

    eval_dir = os.path.join(test_args.model_dir,'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
  
    env_config.reward.base_collision_penalty = -20

    env_config.render_traj = test_args.render_traj
    env_config.save_slides = test_args.save_slides
    env_config.save_path = os.path.join(test_args.model_dir, 'social_eval', test_args.test_model[:-3])
    env_config.args = algo_args

    envs = make_vec_envs(env_name, current_seed, 1,
                         algo_args.gamma, eval_dir, device, allow_early_resets=True,
                         config=env_config, ax=ax, test_case=test_args.test_case, pretext_wrapper=config.env.use_wrapper)

    if config.robot.policy not in ['orca', 'social_force']:
        # load the policy weights
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            env_config,
            base_kwargs=algo_args,
            base=config.robot.policy)
        
        state_dict = torch.load(load_path, map_location=device)
        
        if config.robot.policy == 'networks':
            # Map standard keys to LoRA and rename old module names to new ones
            new_state_dict = {}
            model_state_dict = actor_critic.state_dict()
            
            # Translation map for old variable names to new ones
            name_translation = {
                "humanNodeRNN": "node_rnn",
                "attn": "hr_attn",
                "spatial_attn": "hh_attn",
                "spatial_linear": "hh_out_proj",
                "temporal_edge_layer": "robot_feature_proj",
                "spatial_edge_layer": "human_feature_proj"
            }

            for key, value in state_dict.items():
                translated_key = key
                # 1. Translate old module names to new ones recursively
                for old_name, new_name in name_translation.items():
                    if f".{old_name}." in translated_key:
                        translated_key = translated_key.replace(f".{old_name}.", f".{new_name}.")
                    elif translated_key.startswith(f"{old_name}."):
                        translated_key = translated_key.replace(f"{old_name}.", f"{new_name}.", 1)

                # 2. Map standard Linear to LoRA base_layer if LoRA is enabled
                if hasattr(env_config, 'lora') and env_config.lora.use_lora:
                    base_layer_key = translated_key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                    if base_layer_key in model_state_dict:
                        new_state_dict[base_layer_key] = value
                        continue
                
                new_state_dict[translated_key] = value
            actor_critic.load_state_dict(new_state_dict, strict=False)
        else:
            actor_critic.load_state_dict(state_dict)

        actor_critic.base.nenv = 1
        
        # Apply dynamic LoRA scale
        from rl.networks.network_utils import LoRALinear
        for module in actor_critic.modules():
            if isinstance(module, LoRALinear):
                module.dynamic_scale = test_args.lora_scale
                logging.info(f"Set dynamic_scale for LoRALinear module to {test_args.lora_scale}")

        # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
        nn.DataParallel(actor_critic).to(device)
    else:
        actor_critic = None

    test_size = 250#config.env.test_size
    # call the evaluation function
    evaluate(actor_critic, envs, 1, device, test_size, logging, config, algo_args, model_dir_temp, test_args.visualize, test_args)


if __name__ == '__main__':
    main()