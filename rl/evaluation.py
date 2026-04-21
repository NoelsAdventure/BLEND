import json
import numpy as np
import torch
import os
import csv
from datetime import datetime
import uuid

from crowd_sim.envs.utils.info import *


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, model_dir, visualize=False, test_args=None):
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
        while not done:
            stepCounter = stepCounter + 1
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
            'avg_uncertainty': float(avg_uncertainty)
        })

        print(f"current SR: {success/(k+1)}; current CR: {collision/(k+1)}")

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
    
    # JSON logic
    json_file_path = os.path.join(model_dir, 'test', 'all_evaluations.json')
    all_data = {}
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                all_data = json.load(f)
        except Exception as e:
            logging.error(f"Error reading existing JSON: {e}")
            all_data = {}

    lora_scale = getattr(test_args, 'lora_scale', 1.0)
    exp_id = getattr(test_args, 'exp_id', None)
    if exp_id is None:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    new_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'robot_visible': config.robot.visible,
            'human_random': config.env.randomize_attributes,
            'use_lora': getattr(config.lora, 'use_lora', False),
            'lora_alpha': getattr(config.lora, 'alpha', None),
            'lora_rank': getattr(config.lora, 'rank', None),
            'lora_scale': lora_scale,
            'test_model': getattr(test_args, 'test_model', None),
            'model_dir': model_dir
        },
        'summary': {
            'success_rate': float(success_rate),
            'collision_rate': float(collision_rate),
            'timeout_rate': float(timeout_rate),
            'avg_nav_time': float(avg_nav_time),
            'avg_path_length': float(np.mean(all_path_len)),
            'avg_uncertainty': float(np.mean(all_avg_uncertainty))
        },
        'episodes': episodes_data
    }
    
    all_data[exp_id] = new_entry
    
    with open(json_file_path, 'w') as f:
        json.dump(all_data, f, indent=4)
    
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