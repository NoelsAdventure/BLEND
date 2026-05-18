import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import wandb
from tqdm import tqdm

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy
from rl.networks.storage_safe import RolloutStorage

from crowd_nav.configs.config import Config

def main():
    """
    Main function for training a robot policy network using PPO with cost constraints
    for crowd navigation environments.
    """
    # Read command line arguments and environment configuration
    algo_args = get_args()
    
    if not algo_args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    
    env_config = config = Config()
    
    # Create unique model name based on configuration parameters
    if hasattr(env_config, 'lora') and env_config.lora.use_lora:
        # New LoRA naming convention: use config.note and append LoRA details
        model_name = f"{env_config.note}_rank_{env_config.lora.rank}"
    else:
        # Keep original naming for standard models
        model_name = f"{env_config.note}"
        
    env_config.model_name = model_name
    algo_args.output_dir = f"trained_models/{model_name}"
    
    # Create directory structure for saving training outputs
    if not os.path.exists(algo_args.output_dir):
        os.makedirs(algo_args.output_dir)
    # Prevent accidental overwriting of existing models
    elif not algo_args.overwrite:
        raise ValueError('output_dir already exists!')

    # Save configuration files for reproducibility
    save_config_dir = os.path.join(algo_args.output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy('crowd_nav/configs/config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
    shutil.copy('arguments.py', algo_args.output_dir)

    # Set random seeds for reproducibility
    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            # Reproducible but slower execution
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # Not reproducible but faster execution
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    # Configure PyTorch threading and device
    torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    env_name = algo_args.env_name

    # Special configuration for rendering mode
    if config.sim.render:
        algo_args.num_processes = 1
        algo_args.num_mini_batch = 1
    
    env_config.env.num_processes = algo_args.num_processes
    env_config.args = algo_args

    # Create vectorized environment for parallel training
    envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
                         algo_args.gamma, None, device, False,
                         config=env_config, ax=None,
                         pretext_wrapper=config.env.use_wrapper)

    # Create main policy network (actor-critic)
    actor_critic = Policy(
        envs.observation_space.spaces,  # Dict observation space
        envs.action_space,
        env_config,
        base_kwargs=algo_args,
        base=config.robot.policy)
    
    # Create cost critic network for constrained RL (Both now use env_config with LoRA)
    cost_actor_critic = Policy(
        envs.observation_space.spaces,
        envs.action_space,
        env_config,
        base_kwargs=algo_args,
        base=config.robot.policy)

    # Function to calculate and print parameters
    def print_parameter_count(model, name):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{name}] Total Parameters: {total_params:,}")
        print(f"[{name}] Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    print_parameter_count(actor_critic, "Actor-Critic")
    print_parameter_count(cost_actor_critic, "Cost-Critic")

    # Initialize rollout storage buffer for collecting experience
    rollouts = RolloutStorage(algo_args.num_steps,
                              algo_args.num_processes,
                              envs.observation_space.spaces,
                              envs.action_space,
                              algo_args.human_node_rnn_size,
                              algo_args.human_human_edge_rnn_size)

    # Resume training from checkpoint if specified
    if algo_args.resume:
        load_path = algo_args.load_path
        print(f"Loading weights from {load_path}")
        state_dict = torch.load(load_path, map_location=device, weights_only=False)
        
        # Map standard Linear to LoRA base_layer if LoRA is enabled
        if hasattr(env_config, 'lora') and env_config.lora.use_lora:
            new_state_dict = {}
            model_state_dict = actor_critic.state_dict()
            for key, value in state_dict.items():
                base_layer_key = key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                if base_layer_key in model_state_dict:
                    new_state_dict[base_layer_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # Load weights into both networks
        actor_critic.load_state_dict(state_dict, strict=False)
        cost_actor_critic.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully into both networks.")

    # Explicitly freeze all parameters except LoRA before passing to optimizer
    if hasattr(env_config, 'lora') and getattr(env_config.lora, 'use_lora', False):
        for model in [actor_critic, cost_actor_critic]:
            trainable_params = []
            for name, param in model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    trainable_params.append(name)
            print(f"LoRA mode: {len(trainable_params)} parameters are trainable in {model.__class__.__name__}")

    # Move networks to GPU if available
    nn.DataParallel(actor_critic).to(device)
    nn.DataParallel(cost_actor_critic).to(device)

    # Initialize PPO optimizer with or without cost constraints
    if env_config.policy.constrain_cost:
        # PPO with Lagrangian multipliers for cost constraints
        agent = ppo.PPOLag(
            actor_critic,
            cost_actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            cost_limit=env_config.constrained_rl_related.cost_limit,
            lag_init=env_config.constrained_rl_related.lag_init, 
            lag_lr=env_config.constrained_rl_related.lag_lr,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm,
            )
    else:
        # Standard PPO without cost constraints
        agent = ppo.PPOLagOriginalUpdate(
            actor_critic,
            cost_actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm)
        

    # Initialize Weights & Biases for experiment tracking
    trial_name = model_name
    if algo_args.use_wandb:
        wandb.init(project="robot_crowd_navigation", name=trial_name)
        # Define custom metrics for logging
        wandb.define_metric("env_step")
        wandb.define_metric("train_step")
        wandb.define_metric("env/*", step_metric="env_step")
        wandb.define_metric("train/*", step_metric="train_step")
    
    # Save experiment name and detailed config to files
    note_file_path = os.path.join(algo_args.output_dir, 'note.txt')
    with open(note_file_path, 'w') as file:
        file.write(trial_name)

    summary_file_path = os.path.join(algo_args.output_dir, 'training_summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write(f"Date: {time.ctime()}\n")
        f.write(f"Note: {env_config.note}\n")
        f.write(f"Trial Name: {trial_name}\n")
        f.write(f"Environment: {algo_args.env_name}\n")
        f.write(f"Robot Visible: {env_config.robot.visible}\n")
        
        use_lora = getattr(env_config.lora, 'use_lora', False)
        f.write(f"Use LoRA: {use_lora}\n")
        if use_lora:
            f.write(f"LoRA Rank: {env_config.lora.rank}\n")
            f.write(f"LoRA Alpha: {env_config.lora.alpha}\n")
            
        base_model = algo_args.load_path if algo_args.resume else "Randomly Initialized"
        f.write(f"Base Model: {base_model}\n")
        
        # Add parameter statistics
        for model, name in [(actor_critic, "Actor-Critic"), (cost_actor_critic, "Cost-Critic")]:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"[{name}] Total Parameters: {total_params:,}\n")
            f.write(f"[{name}] Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)\n")

    # Initialize tracking variables
    step_iter = 0
    env_iter = 0
    train_iter = 0

    # Reset environment and get initial observations
    obs = envs.reset()
    # Get ACI (Adaptive Conformity Index) predictions from environment
    out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()
    outs = envs.talk2Env(out_pred)
    aci_predicted_conformity_scores = np.array([o[0] for o in outs])  # [num_envs, num_humans, num_pred_steps]
    aci_cost = np.array([o[1] for o in outs])  # [num_envs,]
    obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(device)
    
    # Initialize rollout storage with first observations
    if isinstance(obs, dict):
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)

    rollouts.to(device)

    # Initialize tracking queues for episode statistics
    episode_rewards = deque(maxlen=500)
    episode_collisions = deque(maxlen=500)
    episode_success = deque(maxlen=500)
    episode_path_length = deque(maxlen=500)
    episode_costs = deque(maxlen=500)
    episode_costs_for_updating_lagrange = deque(maxlen=32)  # For Lagrange multiplier updates
    episode_rewards_for_showing_rewards = deque(maxlen=32)
    best_score = -10000

    # For path length calculation
    current_episode_path_length = np.zeros(algo_args.num_processes)
    # Robustly extract px, py (first two features) regardless of extra dimensions (like sequence length)
    last_robot_pos = obs['robot_node'][..., 0:2].reshape(algo_args.num_processes, 2).cpu().numpy()

    start = time.time()
    total_sim_time = 0.0
    total_train_time = 0.0
    # Calculate total number of training updates
    num_updates = int(algo_args.num_env_steps) // algo_args.num_steps \
                                               // algo_args.num_processes

    # Main training loop
    pbar = tqdm(range(num_updates), desc=model_name)
    for j in pbar:
        # Schedule learning rate decay if enabled
        if algo_args.use_linear_lr_decay:
            network_utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if algo_args.algo == "acktr" else
                algo_args.lr)

        # Collect experience for num_steps timesteps
        sim_start = time.time()
        for step in range(algo_args.num_steps):
            # Sample actions from current policy
            with torch.no_grad():
                # Prepare observations for both main and cost networks
                rollouts_obs = {}
                roullouts_obs_for_cost = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][step]
                    roullouts_obs_for_cost[key] = rollouts.obs[key][step].clone()
                
                # Prepare hidden states for RNN networks
                rollouts_hidden_s = {}
                rollouts_hidden_s_for_cost = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = \
                        rollouts.recurrent_hidden_states[key][step]
                    rollouts_hidden_s_for_cost[key] = \
                        rollouts.recurrent_hidden_states[key][step].clone()
                
                # Get action from main policy
                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(rollouts_obs,
                                     rollouts_hidden_s,
                                     rollouts.masks[step])
                
                # Get cost value from cost critic
                cost_value, _, _, _ = \
                    cost_actor_critic.act(roullouts_obs_for_cost,
                                          rollouts_hidden_s_for_cost,
                                          rollouts.masks[step].clone())

            # Render environment if enabled
            if config.sim.render:
                envs.render()
            
            # Take action and observe results
            obs, reward, done, infos = envs.step(action)
            
            # Path length tracking
            # Ensure it is (num_processes, 2) even if obs has extra dimensions
            current_robot_pos = obs['robot_node'][..., 0:2].view(algo_args.num_processes, 2).cpu().numpy()
            dist = np.linalg.norm(current_robot_pos - last_robot_pos, axis=1)
            current_episode_path_length += dist
            last_robot_pos = current_robot_pos

            # Get ACI predictions and add noise for robustness
            out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()
            outs = envs.talk2Env(out_pred)
            aci_predicted_conformity_scores = np.array([o[0] for o in outs])
            aci_cost = np.array([o[1] for o in outs])
            
            # Add Gaussian noise to conformity scores for data augmentation
            std_dev = env_config.aci_related.noise_std_for_conformity_scores
            gaussian_noise_predicted_conformity_scores = np.random.normal(0.0, std_dev, aci_predicted_conformity_scores.shape)

            # Clip noise to prevent extreme values
            noise_clip_for_c_s = env_config.aci_related.noise_clip_for_conformity_scores
            clipped_noise_predicted_conformity_scores = np.clip(gaussian_noise_predicted_conformity_scores, -noise_clip_for_c_s, noise_clip_for_c_s)

            # Apply noise to conformity scores
            aci_predicted_conformity_scores = aci_predicted_conformity_scores + clipped_noise_predicted_conformity_scores
            
            obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(device)
            
            # Add ACI cost to episode cost tracking
            for i, info in enumerate(infos):
                info['cost'] += aci_cost[i]
            
            # Update environment monitor with new observations
            obs, reward, done, infos = envs.update_monitor(({key: obs[key].cpu().numpy() for key in obs}, reward.numpy(), done, infos))
            processed_costs = torch.tensor([[infos[i]['cost']] for i in range(len(infos))])
            
            # Process episode completion and logging
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    # Track episode statistics
                    episode_rewards.append(info['episode']['r'])
                    episode_costs.append(info['episode']['c'])
                    episode_costs_for_updating_lagrange.append(info['episode']['c'])
                    episode_rewards_for_showing_rewards.append(info['episode']['r'])
                    
                    episode_path_length.append(current_episode_path_length[i])
                    current_episode_path_length[i] = 0

                    # Track success/collision
                    if str(info['info']) == 'Collision':
                        episode_collisions.append(1.0)
                        episode_success.append(0.0)
                    elif str(info['info']) == 'Reaching goal':
                        episode_collisions.append(0.0)
                        episode_success.append(1.0)
                    else:
                        episode_collisions.append(0.0)
                        episode_success.append(0.0)

                    # Save best model based on recent performance
                    mean_num_for_saving = 200
                    avg_score = np.mean(list(episode_rewards)[int(-1*mean_num_for_saving):])
                    if avg_score > best_score:
                        best_score = avg_score
                        
                        # Save main policy
                        save_path_best = os.path.join(algo_args.output_dir, 'best_model')
                        if not os.path.exists(save_path_best):
                            os.mkdir(save_path_best)
                        
                        # Save cost policy
                        cost_save_path_best = os.path.join(algo_args.output_dir, 'cost_best_model')
                        if not os.path.exists(cost_save_path_best):
                            os.mkdir(cost_save_path_best)

                        torch.save(actor_critic.state_dict(),
                                   os.path.join(save_path_best, 'PPO' + ".pt"))
                        torch.save(cost_actor_critic.state_dict(),
                                   os.path.join(cost_save_path_best, 'PPO_cost' + ".pt"))
                
                    # Log environment metrics to wandb
                    if wandb.run:
                        env_iter += 1
                        wandb.log({
                            "env_step": env_iter,
                            "env/Collision": 1 if str(info['info']) == 'Collision' else 0,
                            "env/ReachGoal": 1 if str(info['info']) == 'Reaching goal' else 0,
                            "env/Timeout": 1 if str(info['info']) == 'Timeout' else 0,
                            "env/Episode_Rewards": info['episode']['r'],
                            "env/Episode_Costs": info['episode']['c'],
                            "env/Success_Rate": np.mean(episode_success),
                            "env/Path_Length": episode_path_length[-1] if len(episode_path_length)>0 else 0
                        })
                                        
            # Create masks for episode termination handling
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            
            # Store experience in rollout buffer
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, cost_value, reward, processed_costs, masks, bad_masks)
        
        sim_end = time.time()
        total_sim_time += (sim_end - sim_start)

        # Compute value estimates for the last state
        with torch.no_grad():
            rollouts_obs = {}
            for key in rollouts.obs:
                rollouts_obs[key] = rollouts.obs[key][-1]
            rollouts_hidden_s = {}
            for key in rollouts.recurrent_hidden_states:
                rollouts_hidden_s[key] = \
                    rollouts.recurrent_hidden_states[key][-1]
            
            # Get final value estimates
            next_value = actor_critic.get_value(
                rollouts_obs, rollouts_hidden_s,
                rollouts.masks[-1]).detach()
            cost_next_value = cost_actor_critic.get_value(
                rollouts_obs, rollouts_hidden_s,
                rollouts.masks[-1]).detach()

        # Compute returns and advantages using GAE
        train_start = time.time()
        rollouts.compute_returns(next_value,
                                 cost_next_value,
                                 algo_args.use_gae,
                                 algo_args.gamma,
                                 algo_args.gae_lambda,
                                 algo_args.use_proper_time_limits)

        # Perform policy update
        if len(episode_costs_for_updating_lagrange) > 0:
            mean_ep_costs = np.mean(np.array(episode_costs_for_updating_lagrange))
        else:
            mean_ep_costs = 0.0
            
        if len(episode_rewards_for_showing_rewards) > 0:
            mean_ep_rewards = np.mean(np.array(episode_rewards_for_showing_rewards))
        else:
            mean_ep_rewards = 0.0
            
        value_loss, cost_value_loss, lag_factor, action_loss, dist_entropy, adv_targ_epoch, cost_adv_targ_epoch = agent.update(rollouts, mean_ep_costs)

        rollouts.after_update()
        train_end = time.time()
        total_train_time += (train_end - train_start)
        
        # Update progress bar postfix with latest metrics
        sr = np.mean(episode_success) if len(episode_success) > 0 else 0
        cr = np.mean(episode_collisions) if len(episode_collisions) > 0 else 0
        pl = np.mean(episode_path_length) if len(episode_path_length) > 0 else 0
        rew = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
        cost = np.mean(episode_costs) if len(episode_costs) > 0 else 0
        pbar.set_postfix({
            'SR': f'{sr:.2f}',
            'CR': f'{cr:.2f}',
            'PL': f'{pl:.2f}',
            'Rew': f'{rew:.2f}',
            'Cost': f'{cost:.2f}'
        })

        # Log training metrics
        if algo_args.use_wandb:
            train_iter += 1
            wandb.log({
                "train_step": train_iter,
                "train/value_loss": value_loss,
                "train/cost_value_loss": cost_value_loss,
                "train/lag_factor": lag_factor,
                "train/action_loss": action_loss,
                "train/dist_entropy": dist_entropy,
                "train/adv_targ_epoch": adv_targ_epoch,
                "train/cost_adv_targ_epoch": cost_adv_targ_epoch,
                "train/mean_ep_costs": mean_ep_costs,
                "train/mean_ep_rewards": mean_ep_rewards,
                "train/sim_time": total_sim_time,
                "train/update_time": total_train_time
            })

        # Save model checkpoints periodically
        if j % algo_args.save_interval == 0 or j == num_updates - 1:
            # Save main policy checkpoints
            save_path = os.path.join(algo_args.output_dir, 'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            # Save cost policy checkpoints
            cost_save_path = os.path.join(algo_args.output_dir, 'cost_checkpoints')
            if not os.path.exists(cost_save_path):
                os.mkdir(cost_save_path)

            torch.save(actor_critic.state_dict(),
                       os.path.join(save_path, '%.5i' % j + ".pt"))
            
            torch.save(cost_actor_critic.state_dict(),
                       os.path.join(cost_save_path, '%.5i' % j + ".pt"))

        # Print training progress
        if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
            end = time.time()

            pbar.write(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Total Sim Time: {:.2f}s, Total Train Time: {:.2f}s\n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "mean/median cost {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    total_sim_time,
                    total_train_time,
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.mean(episode_costs),
                    np.median(episode_costs),
                    np.min(episode_rewards),
                    np.max(episode_rewards)
                )
            )
            pbar.write(f"Collision rate: {np.mean(episode_collisions):.2f}, Success rate: {np.mean(episode_success):.2f}, Path length: {np.mean(episode_path_length):.2f}")

            # Save training progress to CSV
            df = pd.DataFrame({'misc/nupdates': [j],
                               'misc/total_timesteps': [total_num_steps],
                               'fps': int(total_num_steps / (end - start)),
                               'sim_time': [total_sim_time],
                               'train_time': [total_train_time],
                               'eprewmean': [np.mean(episode_rewards)],
                               'epcostmean': [np.mean(episode_costs)],
                               'epsuccessmean': [np.mean(episode_success)],
                               'epcollisionmean': [np.mean(episode_collisions)],
                               'eppathlengthmean': [np.mean(episode_path_length)],
                               'loss/policy_entropy': dist_entropy,
                               'loss/policy_loss': action_loss,
                               'loss/value_loss': value_loss,
                               'loss/cost_value_loss': cost_value_loss})

            if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
                df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'),
                          mode='a', header=False, index=False)
            else:
                df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'),
                          mode='w', header=True, index=False)

    envs.close()
    
if __name__ == '__main__':
    main()
