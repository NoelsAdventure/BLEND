import json
import numpy as np
import torch
import os
import csv
from datetime import datetime
import uuid

from crowd_sim.envs.utils.info import *


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, model_dir, visualize=False, test_args=None, video_save_path=None):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []
    all_avg_uncertainty = []

    # Store detailed per-episode data
    episodes_data = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

        # Experiment ID logic (moved up for video saving)
    exp_id = getattr(test_args, 'exp_id', None)
    if exp_id is None:
        scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
        behaviour = getattr(test_args, 'lora_behaviour', 'none')
        if scenario != 'none' or behaviour != 'none':
            exp_id = f"{scenario}_{behaviour}"
        else:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
        outs = baseEnv.talk2Env(out_pred)
        aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
        
        # Track uncertainty for the episode
        episode_uncertainties = []
        if aci_predicted_conformity_scores is not None and len(aci_predicted_conformity_scores) > 0:
            episode_uncertainties.append(np.mean(aci_predicted_conformity_scores))

        aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
        obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

        if config.robot.policy not in ['orca', 'social_force']:
            eval_recurrent_hidden_states = {}

            node_num = 1
            edge_num = actor_critic.base.human_num + 1
            eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                        device=device)

            eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                            actor_critic.base.human_human_edge_rnn_size,
                                                                            device=device)
        
        # Adaptive LoRA Proof of Concept: Initialization
        scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
        behaviour = getattr(test_args, 'lora_behaviour', 'none')
        
        # Default lora_scale from args
        baseEnv.robot.lora_scale = getattr(test_args, 'lora_scale', 1.0)

        # Clean up any leftover PoC attributes from previous episodes
        if hasattr(baseEnv.robot, 'visible_to_humans'):
            delattr(baseEnv.robot, 'visible_to_humans')

        # Behaviour Overrides
        if behaviour == 'always_on':
            baseEnv.robot.lora_enabled = True
            baseEnv.robot.lora_scale = 1.0
            from rl.networks.network_utils import LoRALinear, LoRAAdapter
            for module in actor_critic.modules():
                if isinstance(module, (LoRALinear, LoRAAdapter)):
                    module.dynamic_scale = 1.0
        elif behaviour == 'always_off':
            baseEnv.robot.lora_enabled = False
            baseEnv.robot.lora_scale = 0.0
            from rl.networks.network_utils import LoRALinear, LoRAAdapter
            for module in actor_critic.modules():
                if isinstance(module, (LoRALinear, LoRAAdapter)):
                    module.dynamic_scale = 0.0
            
        if scenario == 'seperate_mixed_5050':
            # One group (half) is ignorant (False), the other is friendly (True)
            baseEnv.robot.visible_to_humans = [False if i < len(baseEnv.humans)//2 else True 
                                               for i in range(len(baseEnv.humans))]
            print(f"Group scenario: {sum(not v for v in baseEnv.robot.visible_to_humans)} ignorant, {sum(baseEnv.robot.visible_to_humans)} friendly")
            
            # For visualization of baseline models in group scenario
            baseEnv.robot.lora_enabled = "lora" in model_dir.lower() or "alpha" in model_dir.lower()

        elif scenario == 'seperate_ignorant_to_aware_step25':
            # Start invisible (Ignorant humans)
            baseEnv.robot.visible = False
            
            # Start with LoRA OFF (if switching)
            if behaviour in ['switching', 'adaptive']:
                baseEnv.robot.lora_enabled = False
                baseEnv.robot.lora_scale = 0.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 0.0

        elif scenario == 'seperate_all_ignorant':
            baseEnv.robot.visible = False
            if behaviour in ['switching', 'adaptive']:
                baseEnv.robot.lora_enabled = False
                baseEnv.robot.lora_scale = 0.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 0.0

        elif scenario == 'seperate_all_aware':
            baseEnv.robot.visible = True
            if behaviour in ['switching', 'adaptive']:
                baseEnv.robot.lora_enabled = True
                baseEnv.robot.lora_scale = 1.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 1.0

        episode_steps = []
        while not done:
            stepCounter = stepCounter + 1
            
            # Adaptive LoRA Proof of Concept: Mid-episode switch for 'seperate_ignorant_to_aware_step25' scenario
            if scenario == 'seperate_ignorant_to_aware_step25' and stepCounter == 25:
                baseEnv.robot.visible = True
                if behaviour in ['switching', 'adaptive']:
                    print(f"\n>>> Step {stepCounter}: Switching to VISIBLE and LoRA ON (Adaptive PoC)")
                    baseEnv.robot.lora_scale = 1.0
                        
                    from rl.networks.network_utils import LoRALinear, LoRAAdapter
                    for module in actor_critic.modules():
                        if isinstance(module, (LoRALinear, LoRAAdapter)):
                            module.dynamic_scale = 1.0

            # Adaptive LoRA Proof of Concept: Zone-based switching for 'seperate_mixed_5050' scenario
            if scenario == 'seperate_mixed_5050':
                if behaviour == 'adaptive':
                    # Continuous adaptive scale based on distance-weighted ratio
                    robot_pos = baseEnv.robot.get_position()
                    total_weight = 0.0
                    friendly_weight = 0.0
                    
                    for i, human in enumerate(baseEnv.humans):
                        dist = np.linalg.norm(np.array(human.get_position()) - np.array(robot_pos))
                        if dist <= baseEnv.robot.sensor_range:
                            # Calculate weight: closer humans have higher weight (1.0 at dist=0, 0.0 at dist=sensor_range)
                            weight = max(0.0, 1.0 - (dist / baseEnv.robot.sensor_range))
                            total_weight += weight
                            
                            if baseEnv.robot.visible_to_humans[i]:
                                friendly_weight += weight
                    
                    # Calculate continuous target scale
                    if total_weight <= 0.0:
                        target_scale = 1.0  # Default to 1.0 if no humans in range
                    else:
                        target_scale = friendly_weight / total_weight
                        
                    # Apply change if scale differs significantly
                    if abs(target_scale - baseEnv.robot.lora_scale) > 0.01:
                        print(f"\n>>> Step {stepCounter}: Target Scale = {target_scale:.2f} (Total Weight: {total_weight:.2f}, Friendly: {friendly_weight:.2f})")
                        baseEnv.robot.lora_scale = target_scale
                        baseEnv.robot.lora_enabled = (target_scale > 0)
                        
                        from rl.networks.network_utils import LoRALinear, LoRAAdapter
                        for module in actor_critic.modules():
                            if isinstance(module, (LoRALinear, LoRAAdapter)):
                                module.dynamic_scale = target_scale

                elif behaviour == 'switching':
                    # Get robot position
                    robot_pos = baseEnv.robot.get_position()
                    
                    humans_in_range = []
                    friendly_in_range = 0
                    
                    for i, human in enumerate(baseEnv.humans):
                        dist = np.linalg.norm(np.array(human.get_position()) - np.array(robot_pos))
                        if dist <= baseEnv.robot.sensor_range:
                            humans_in_range.append(i)
                            if baseEnv.robot.visible_to_humans[i]:
                                friendly_in_range += 1
                    
                    # Calculate target lora_scale
                    if not humans_in_range:
                        target_scale = 1.0  # Default to LoRA ON if no humans in range
                        ratio = 1.0
                    else:
                        ratio = friendly_in_range / len(humans_in_range)
                        target_scale = 1.0 if ratio > 0.5 else 0.0
                    
                    # Apply change if scale needs update
                    if target_scale != baseEnv.robot.lora_scale:
                        msg = f"Ratio {friendly_in_range}/{len(humans_in_range)}={ratio:.2f}" if humans_in_range else "No humans in range"
                        print(f"\n>>> Step {stepCounter}: {msg}. Switching LoRA to {target_scale}")
                        baseEnv.robot.lora_scale = target_scale
                        baseEnv.robot.lora_enabled = (target_scale > 0)
                        
                        from rl.networks.network_utils import LoRALinear, LoRAAdapter
                        for module in actor_critic.modules():
                            if isinstance(module, (LoRALinear, LoRAAdapter)):
                                module.dynamic_scale = target_scale
            
            # Collect data for the CURRENT step before taking the next action
            # Robot features in robot_node: [px, py, radius, gx, gy, v_pref, theta]
            # Robot features in temporal_edges: [vx, vy]
            r_node = obs['robot_node'][0, 0].cpu().numpy()
            r_vel = obs['temporal_edges'][0, 0].cpu().numpy()
            
            # Current human states
            human_states = []
            for h in baseEnv.humans:
                human_states.append({
                    'id': int(h.id),
                    'pos': [float(h.px), float(h.py)],
                    'vel': [float(h.vx), float(h.vy)],
                    'radius': float(h.radius)
                })

            step_data = {
                'step': stepCounter,
                'robot': {
                    'pos': [float(r_node[0]), float(r_node[1])],
                    'vel': [float(r_vel[0]), float(r_vel[1])],
                    'goal': [float(r_node[3]), float(r_node[4])],
                    'theta': float(r_node[6]),
                    'radius': float(r_node[2])
                },
                'humans': human_states,
                'pred_traj': out_pred.tolist(),
                'uncertainty': aci_predicted_conformity_scores[0].tolist() if aci_predicted_conformity_scores is not None else None
            }
            episode_steps.append(step_data)

            if config.robot.policy not in ['orca', 'social_force']:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if visualize:
                eval_envs.render()
                if video_save_path:
                    baseEnv.plot_step(video_save_path)

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)
            
            out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
            # send manager action to all processes
            out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
            outs = baseEnv.talk2Env(out_pred)
            aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
            
            # Track uncertainty for each step
            if aci_predicted_conformity_scores is not None and len(aci_predicted_conformity_scores) > 0:
                episode_uncertainties.append(np.mean(aci_predicted_conformity_scores))

            aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
            obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
            # render

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)
        
        avg_uncertainty = np.mean(episode_uncertainties) if len(episode_uncertainties) > 0 else 0.0
        all_avg_uncertainty.append(avg_uncertainty)

        episode_result = 'Unknown'
        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            episode_result = 'Success'
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            episode_result = 'Collision'
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            episode_result = 'Timeout'
            print('Time out')
        
        episodes_data.append({
            'episode': k,
            'result': episode_result,
            'reward': float(episode_rew),
            'steps': stepCounter,
            'time': float(global_time),
            'path_length': float(path_len),
            'avg_uncertainty': float(avg_uncertainty),
            'steps_data': episode_steps
        })

        print(f"current SR: {success/(k+1)}; current CR: {collision/(k+1)}")

        if video_save_path:
            baseEnv.animate_episode(video_save_path, f"{exp_id}_ep{k}_{episode_result}")

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    logging.info(
        'Testing success rate: {:.4f}, collision rate: {:.4f}, timeout rate: {:.4f}, '
        'nav time: {:.4f}, path length: {:.4f}, average intrusion ratio: {:.4f}%, '
        'average minimal distance during intrusions: {:.4f}, average prediction uncertainty: {:.4f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist), np.mean(all_avg_uncertainty)))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    
    # JSON logic: Save a summary to the main file and full data to a unique experiment file
    lora_scale = getattr(test_args, 'lora_scale', 1.0)
    
    # Detailed config for the experiment
    important_config = {
        'robot_visible': config.robot.visible,
        'robot_fov': float(config.robot.FOV),
        'human_num': int(config.sim.human_num),
        'human_random': config.env.randomize_attributes,
        'human_fov': float(config.humans.FOV),
        'env_name': args.env_name,
        'use_lora': getattr(getattr(config, 'lora', object()), 'use_lora', False),
        'lora_alpha': getattr(getattr(config, 'lora', object()), 'alpha', None),
        'lora_rank': getattr(getattr(config, 'lora', object()), 'rank', None),
        'lora_scale': lora_scale,
        'test_model': getattr(test_args, 'test_model', None),
        'model_dir': model_dir
    }

    full_experiment_data = {
        'exp_id': exp_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': important_config,
        'summary': {
            'num_episodes': int(test_size),
            'success_rate': float(success_rate),
            'collision_rate': float(collision_rate),
            'timeout_rate': float(timeout_rate),
            'avg_nav_time': float(avg_nav_time),
            'avg_path_length': float(np.mean(all_path_len)),
            'avg_uncertainty': float(np.mean(all_avg_uncertainty))
        },
        'episodes': episodes_data
    }
    
    # 1. Save FULL data for THIS experiment
    individual_json_path = os.path.join(model_dir, 'test', f'{exp_id}.json')
    with open(individual_json_path, 'w') as f:
        json.dump(full_experiment_data, f, indent=4)
    logging.info(f"Full experiment data saved to {individual_json_path}")

    # 2. Update/Create summary index of ALL experiments
    summary_json_path = os.path.join(model_dir, 'test', 'all_evaluations.json')
    all_summaries = {}
    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, 'r') as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = {}

    # Create a lightweight entry for the master index
    summary_entry = full_experiment_data.copy()
    del summary_entry['episodes'] # Remove heavy data for the index
    summary_entry['data_file'] = f'{exp_id}.json'
    
    all_summaries[exp_id] = summary_entry
    
    with open(summary_json_path, 'w') as f:
        json.dump(all_summaries, f, indent=4)
    
    # Keep CSV for backward compatibility (per-run)
    csv_file_path = os.path.join(model_dir, 'test', f'evaluation_data_scale_{lora_scale}.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Success Times', 'Collision Times', 'Timeout Times', 'Path Length', 'Min Distance', 'Avg Uncertainty'])
        
        max_length = max(len(success_times), len(collision_times), len(timeout_times), len(all_path_len), len(min_dist), len(all_avg_uncertainty))
        for i in range(max_length):
            row = [
                success_times[i] if i < len(success_times) else '',
                collision_times[i] if i < len(collision_times) else '',
                timeout_times[i] if i < len(timeout_times) else '',
                all_path_len[i] if i < len(all_path_len) else '',
                min_dist[i] if i < len(min_dist) else '',
                all_avg_uncertainty[i] if i < len(all_avg_uncertainty) else ''
            ]
            writer.writerow(row)
    eval_envs.close()