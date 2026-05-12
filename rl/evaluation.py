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
    
    # Awareness prediction accuracy tracking
    total_awareness_predictions = 0
    correct_awareness_predictions = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    all_discrepancy_data = {'aware': [], 'ignorant': []}

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
    user_exp_id = getattr(test_args, 'exp_id', None)
    scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
    behaviour = getattr(test_args, 'lora_behaviour', 'none')
    
    if scenario != 'none' or behaviour != 'none':
        base_id = f"{scenario}_{behaviour}"
        if user_exp_id:
            exp_id = f"{base_id}_exp{user_exp_id}"
        else:
            exp_id = base_id
    else:
        if user_exp_id:
            exp_id = f"exp{user_exp_id}"
        else:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    # start the testing episodes
    try:
        from tqdm import tqdm
        pbar = tqdm(range(test_size), desc='Evaluating')
    except ImportError:
        pbar = range(test_size)

    for k in pbar:
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
        elif behaviour == 'fixed_scale':
            scale = getattr(test_args, 'lora_scale', 1.0)
            baseEnv.robot.lora_enabled = (scale > 0)
            baseEnv.robot.lora_scale = scale
            from rl.networks.network_utils import LoRALinear, LoRAAdapter
            for module in actor_critic.modules():
                if isinstance(module, (LoRALinear, LoRAAdapter)):
                    module.dynamic_scale = scale
            
        if scenario == 'seperate_mixed_5050':
            # One group (half) is ignorant (False), the other is friendly (True)
            baseEnv.robot.visible_to_humans = [False if i < len(baseEnv.humans)//2 else True 
                                               for i in range(len(baseEnv.humans))]
            if k == 0:
                msg = f"Group scenario: {sum(not v for v in baseEnv.robot.visible_to_humans)} ignorant, {sum(baseEnv.robot.visible_to_humans)} friendly"
                if hasattr(pbar, 'write'):
                    pbar.write(msg)
                else:
                    print(msg)
            
            # For visualization of baseline models in group scenario
            baseEnv.robot.lora_enabled = "lora" in model_dir.lower() or "alpha" in model_dir.lower()

        elif scenario == 'seperate_ignorant_to_aware_step25':
            # Start invisible (Ignorant humans)
            baseEnv.robot.visible = False
            
            # Start with LoRA OFF (if switching)
            if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew']:
                baseEnv.robot.lora_enabled = False
                baseEnv.robot.lora_scale = 0.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 0.0

        elif scenario == 'seperate_all_ignorant':
            baseEnv.robot.visible = False
            if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew']:
                baseEnv.robot.lora_enabled = False
                baseEnv.robot.lora_scale = 0.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 0.0

        elif scenario == 'seperate_all_aware':
            baseEnv.robot.visible = True
            if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew']:
                baseEnv.robot.lora_enabled = True
                baseEnv.robot.lora_scale = 1.0
                from rl.networks.network_utils import LoRALinear, LoRAAdapter
                for module in actor_critic.modules():
                    if isinstance(module, (LoRALinear, LoRAAdapter)):
                        module.dynamic_scale = 1.0

        # Initialize prev_predictions before the loop for tracking discrepancy
        prev_predictions = {}
        human_above_threshold_count = {} # Track consecutive frames above threshold
        for human in baseEnv.humans:
            if human.last_prediction is not None and len(human.last_prediction) > 1:
                prev_predictions[human.id] = {
                    'past_human': human.last_prediction[0],
                    'past_pred': human.last_prediction[1]
                }

        episode_steps = []
        episode_uncertainties = []
        episode_lora_scales = []
        episode_discrepancies = []
        episode_friendly_flags = []
        while not done:
            stepCounter = stepCounter + 1
            
            # Calculate Prediction Discrepancy Scores to guess who is aware/ignorant
            discrepancy_scores = {}
            robot_pos = baseEnv.robot.get_position()
            
            for i, human in enumerate(baseEnv.humans):
                if human.id in prev_predictions:
                    actual_human_pos = np.array([human.px, human.py])
                    
                    if isinstance(prev_predictions[human.id], dict):
                        predicted_human_pos = prev_predictions[human.id]['past_pred']
                        past_human_pos = prev_predictions[human.id]['past_human']
                    else:
                        predicted_human_pos = prev_predictions[human.id]
                        past_human_pos = None
                    
                    # Calculate vectors
                    v_robot_pred = np.array(robot_pos) - predicted_human_pos
                    v_human_pred = actual_human_pos - predicted_human_pos

                    if 'discrepancynew' in behaviour and past_human_pos is not None:
                        # New Formula: Angle between (past human -> past prediction) and (past prediction -> actual human)
                        v_past_human_to_pred = predicted_human_pos - past_human_pos
                        v_pred_to_actual = actual_human_pos - predicted_human_pos
                        
                        angle_past = np.arctan2(v_past_human_to_pred[1], v_past_human_to_pred[0])
                        angle_actual = np.arctan2(v_pred_to_actual[1], v_pred_to_actual[0])
                        psi = angle_actual - angle_past
                    else:
                        # Original logic: Angle between (robot -> pred) and (human -> pred)
                        angle_robot = np.arctan2(v_robot_pred[1], v_robot_pred[0])
                        angle_human = np.arctan2(v_human_pred[1], v_human_pred[0])
                        psi = angle_human - angle_robot

                    # Calculate discrepancy score: ||P_human - P_pred_before|| * abs(sin(psi))
                    discrepancy_score = np.linalg.norm(v_human_pred) * abs(np.sin(psi))                        
                    discrepancy_scores[human.id] = discrepancy_score

                    # Track ground truth for statistics collection
                    if hasattr(baseEnv.robot, 'visible_to_humans'):
                        actual_friendly = baseEnv.robot.visible_to_humans[i]
                        if actual_friendly:
                            all_discrepancy_data['aware'].append(float(discrepancy_score))
                        else:
                            all_discrepancy_data['ignorant'].append(float(discrepancy_score))
                    else:
                        # If visibility not set per human, check overall robot.visible
                        if baseEnv.robot.visible:
                            all_discrepancy_data['aware'].append(float(discrepancy_score))
                        else:
                            all_discrepancy_data['ignorant'].append(float(discrepancy_score))

            # Adaptive LoRA Proof of Concept: Mid-episode switch for 'seperate_ignorant_to_aware_step25' scenario
            if scenario == 'seperate_ignorant_to_aware_step25' and stepCounter == 25:
                baseEnv.robot.visible = True
                if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew']:
                    if test_size <= 10:
                        print(f"\n>>> Step {stepCounter}: Switching to VISIBLE and LoRA ON (Adaptive PoC)")
                    baseEnv.robot.lora_scale = 1.0
                        
                    from rl.networks.network_utils import LoRALinear, LoRAAdapter
                    for module in actor_critic.modules():
                        if isinstance(module, (LoRALinear, LoRAAdapter)):
                            module.dynamic_scale = 1.0

            # Continuous adaptive scale or majority-based switching
            if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew']:
                robot_pos = baseEnv.robot.get_position()
                robot_theta = obs['robot_node'][0, 0, 6].item() # robot heading
                total_weight = 0.0
                friendly_weight = 0.0
                friendly_in_range = 0
                humans_in_range_count = 0
                
                for i, human in enumerate(baseEnv.humans):
                    dist = np.linalg.norm(np.array(human.get_position()) - np.array(robot_pos))
                    if dist <= baseEnv.robot.sensor_range:
                        humans_in_range_count += 1
                        
                        # Awareness Detection
                        if '_gt' in behaviour:
                            # Ground Truth awareness
                            if hasattr(baseEnv.robot, 'visible_to_humans'):
                                is_friendly = baseEnv.robot.visible_to_humans[i]
                            else:
                                is_friendly = baseEnv.robot.visible
                            actual_friendly = is_friendly
                        else:
                            # Discrepancy-based awareness
                            score = discrepancy_scores.get(human.id, 0.0)
                            threshold = getattr(test_args, 'discrepancy_threshold', 0.05)
                            m_threshold = getattr(test_args, 'discrepancy_m', 1)
                            
                            if score > threshold:
                                human_above_threshold_count[human.id] = human_above_threshold_count.get(human.id, 0) + 1
                            else:
                                human_above_threshold_count[human.id] = 0
                                
                            is_friendly = human_above_threshold_count.get(human.id, 0) >= m_threshold
                            
                            # Track accuracy against ground truth for statistics
                            actual_friendly = False
                            if hasattr(baseEnv.robot, 'visible_to_humans'):
                                actual_friendly = baseEnv.robot.visible_to_humans[i]
                            else:
                                actual_friendly = baseEnv.robot.visible
                                
                            total_awareness_predictions += 1
                            if is_friendly == actual_friendly:
                                correct_awareness_predictions += 1
                            
                            if is_friendly and actual_friendly:
                                tp += 1
                            elif is_friendly and not actual_friendly:
                                fp += 1
                            elif not is_friendly and actual_friendly:
                                fn += 1
                            elif not is_friendly and not actual_friendly:
                                tn += 1
                        
                        if is_friendly:
                            friendly_in_range += 1
                            
                        # Calculate distance-based weight (1.0 at dist=0, 0.0 at dist=sensor_range)
                        dist_weight = max(0.0, 1.0 - (dist / baseEnv.robot.sensor_range))
                        
                        # Calculate directional weight (cos of half angle difference)
                        angle_to_human = np.arctan2(human.py - robot_pos[1], human.px - robot_pos[0])
                        theta_i = robot_theta - angle_to_human
                        
                        # Ensure theta_i is in [-pi, pi] for consistent cos(theta/2) behavior
                        theta_i = (theta_i + np.pi) % (2 * np.pi) - np.pi
                        direction_weight = np.cos(theta_i / 2)
                        
                        weight = dist_weight * direction_weight
                        
                        total_weight += weight
                        if is_friendly:
                            friendly_weight += weight
                
                # Calculate target lora_scale
                if humans_in_range_count == 0:
                    target_scale = 0.0  # Default to LoRA OFF if no humans in range
                    ratio = 0.0
                elif 'switching' in behaviour:
                    ratio = friendly_in_range / humans_in_range_count
                    target_scale = 1.0 if ratio > 0.5 else 0.0
                else: # adaptive
                    ratio = friendly_weight / total_weight if total_weight > 0 else 0.0
                    target_scale = ratio
                
                # Apply change if scale differs significantly
                if abs(target_scale - baseEnv.robot.lora_scale) > 0.01:
                    if test_size <= 10:
                        msg = f"Ratio {ratio:.2f}"
                        if 'switching' in behaviour:
                            msg = f"Ratio {friendly_in_range}/{humans_in_range_count}={ratio:.2f}"
                        print(f"\n>>> Step {stepCounter}: {msg}. Target Scale = {target_scale:.2f}")
                    
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
            for i, h in enumerate(baseEnv.humans):
                # Determine if this human was considered "friendly" in the current step
                # (re-calculating awareness here to match the logic used in the scaling step)
                is_friendly = False
                
                # Check if discrepancy is actually available for this human
                h_discrepancy_raw = discrepancy_scores.get(h.id)
                h_discrepancy_for_logic = h_discrepancy_raw if h_discrepancy_raw is not None else 0.0
                if behaviour in ['switching_gt', 'adaptive_gt']:
                    if hasattr(baseEnv.robot, 'visible_to_humans'):
                        is_friendly = bool(baseEnv.robot.visible_to_humans[i])
                    else:
                        is_friendly = bool(baseEnv.robot.visible)
                elif behaviour in ['switching_discrepancy', 'adaptive_discrepancy', 'switching_discrepancynew', 'adaptive_discrepancynew']:
                    threshold = getattr(test_args, 'discrepancy_threshold', 0.05)
                    is_friendly = h_discrepancy_for_logic > threshold
                
                human_states.append({
                    'id': int(h.id),
                    'pos': [float(h.px), float(h.py)],
                    'vel': [float(h.vx), float(h.vy)],
                    'radius': float(h.radius),
                    'discrepancy': float(h_discrepancy_raw) if h_discrepancy_raw is not None else None,
                    'is_friendly': bool(is_friendly)
                })
                
                # Only include in averages if the data was actually available
                if h_discrepancy_raw is not None:
                    episode_discrepancies.append(float(h_discrepancy_raw))
                
                episode_friendly_flags.append(bool(is_friendly))

            step_data = {
                'step': stepCounter,
                'robot': {
                    'pos': [float(r_node[0]), float(r_node[1])],
                    'vel': [float(r_vel[0]), float(r_vel[1])],
                    'goal': [float(r_node[3]), float(r_node[4])],
                    'theta': float(r_node[6]),
                    'radius': float(r_node[2]),
                    'lora_scale': float(baseEnv.robot.lora_scale)
                },
                'humans': human_states,
                'pred_traj': out_pred.tolist(),
                'uncertainty': aci_predicted_conformity_scores[0].tolist() if aci_predicted_conformity_scores is not None else None
            }
            episode_steps.append(step_data)
            episode_lora_scales.append(float(baseEnv.robot.lora_scale))

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
            
            # Update prev_predictions for the next step
            for h in baseEnv.humans:
                if h.last_prediction is not None and len(h.last_prediction) > 1:
                    prev_predictions[h.id] = {
                        'past_human': h.last_prediction[0],
                        'past_pred': h.last_prediction[1]
                    }
            
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
        if visualize:
            print('')
            print('Reward={}'.format(episode_rew))
            print('Episode', k, 'ends in', stepCounter)
        
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)
        
        avg_uncertainty = np.mean(episode_uncertainties) if len(episode_uncertainties) > 0 else 0.0
        all_avg_uncertainty.append(avg_uncertainty)

        avg_lora_scale = np.mean(episode_lora_scales) if len(episode_lora_scales) > 0 else 0.0
        avg_discrepancy_ep = np.mean(episode_discrepancies) if len(episode_discrepancies) > 0 else 0.0
        avg_friendly_ratio_ep = np.mean(episode_friendly_flags) if len(episode_friendly_flags) > 0 else 0.0

        episode_result = 'Unknown'
        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            episode_result = 'Success'
            if visualize: print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            episode_result = 'Collision'
            if visualize: print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            episode_result = 'Timeout'
            if visualize: print('Time out')
        
        episodes_data.append({
            'episode': k,
            'result': episode_result,
            'reward': float(episode_rew),
            'steps': stepCounter,
            'time': float(global_time),
            'path_length': float(path_len),
            'avg_uncertainty': float(avg_uncertainty),
            'avg_lora_scale': float(avg_lora_scale),
            'avg_discrepancy': float(avg_discrepancy_ep),
            'avg_friendly_ratio': float(avg_friendly_ratio_ep),
            'steps_data': episode_steps
        })

        if hasattr(pbar, 'set_postfix'):
            avg_lora_so_far = np.mean([ep['avg_lora_scale'] for ep in episodes_data])
            avg_pl_so_far = np.mean(all_path_len)
            pbar.set_postfix({
                'SR': f'{success/(k+1):.2f}',
                'CR': f'{collision/(k+1):.2f}',
                'Avg PL': f'{avg_pl_so_far:.2f}',
                'Avg LoRA': f'{avg_lora_so_far:.2f}'
            })

        if not visualize and (k + 1) % 50 == 0:
            avg_sr = success / (k + 1)
            avg_cr = collision / (k + 1)
            avg_lora = np.mean([ep['avg_lora_scale'] for ep in episodes_data])
            summary_str = f"[Step {k+1}] SR: {avg_sr:.3f}, CR: {avg_cr:.3f}, Avg LoRA: {avg_lora:.3f}"
            if hasattr(pbar, 'write'):
                pbar.write(summary_str)
            else:
                print(f"\n{summary_str}")

        if video_save_path:
            baseEnv.animate_episode(video_save_path, f"{exp_id}_ep{k}_{episode_result}", outcome=episode_result, avg_lora_scale=avg_lora_scale)

    if not visualize:
        print() # Move to next line after progress bar

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
        'average minimal distance during intrusions: {:.4f}, average prediction uncertainty: {:.4f}, '
        'average LoRA scale: {:.4f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist), np.mean(all_avg_uncertainty),
                   np.mean([ep['avg_lora_scale'] for ep in episodes_data])))

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

    # Calculate average discrepancy scores for the experiment
    aware_scores = all_discrepancy_data.get('aware', [])
    ignorant_scores = all_discrepancy_data.get('ignorant', [])
    all_scores = aware_scores + ignorant_scores
    
    avg_discrepancy_aware = np.mean(aware_scores) if aware_scores else 0.0
    avg_discrepancy_ignorant = np.mean(ignorant_scores) if ignorant_scores else 0.0
    avg_discrepancy_all = np.mean(all_scores) if all_scores else 0.0

    full_experiment_data = {
        'exp_id': exp_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': important_config,
        'summary': {
            'num_episodes': test_size,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_nav_time': avg_nav_time,
            'avg_path_length': float(np.mean(all_path_len)),
            'avg_uncertainty': float(np.mean(all_avg_uncertainty)),
            'avg_lora_scale': float(np.mean([ep['avg_lora_scale'] for ep in episodes_data])),
            'avg_discrepancy_aware': float(avg_discrepancy_aware),
            'avg_discrepancy_ignorant': float(avg_discrepancy_ignorant),
            'avg_discrepancy_all': float(avg_discrepancy_all),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
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
    if total_awareness_predictions > 0:
        accuracy = (correct_awareness_predictions / total_awareness_predictions) * 100
        print(f"\n==========================================================")
        print(f"AWARENESS PREDICTION ACCURACY")
        print(f"==========================================================")
        print(f"Total Predictions: {total_awareness_predictions}")
        print(f"Correct Predictions: {correct_awareness_predictions}")
        print(f"Accuracy: {accuracy:.2f}% (Threshold: {getattr(test_args, 'discrepancy_threshold', 0.05)}, M: {getattr(test_args, 'discrepancy_m', 1)})")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"==========================================================\n")

    # Save discrepancy data for analysis
    data_path = os.path.join(model_dir, 'test', f'discrepancy_data_{scenario}.json')
    with open(data_path, 'w') as f:
        json.dump(all_discrepancy_data, f, indent=4)
    logging.info(f"Discrepancy data saved to {data_path}")

    eval_envs.close()